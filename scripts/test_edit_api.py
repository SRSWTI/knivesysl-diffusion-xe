#!/usr/bin/env python3
"""Real-model editing integration tests. No mocks; requires the AXE service.

python3 scripts/test_edit_api.py --base http://127.0.0.1:8080
Generated, restricted test fixtures are evaluated in a resource-limited child
process. The serving application never executes generated or submitted code.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request

BASE = 'http://127.0.0.1:8080'


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as response:
        return json.load(response)


def post(payload):
    request = urllib.request.Request(BASE + '/edit', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=180)


def events(response):
    for line in response:
        if line.startswith(b'data: '):
            yield json.loads(line[6:])


def evaluate_fixture(source, name, cases):
    result = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), '--evaluate'],
                            input=json.dumps({'source': source, 'name': name, 'cases': cases}),
                            text=True, capture_output=True, timeout=8)
    if result.returncode:
        raise AssertionError('Restricted fixture evaluation failed: ' + result.stderr[-2000:])
    return json.loads(result.stdout)


def evaluator():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    payload = json.load(sys.stdin)
    tree = ast.parse(payload['source'])
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
               ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Compare, ast.Eq,
               ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.BinOp, ast.Add,
               ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
               ast.ListComp, ast.comprehension, ast.Name, ast.Load, ast.Store,
               ast.Constant, ast.List, ast.Tuple, ast.Subscript, ast.Call,
               ast.Expr, ast.Pass, ast.UnaryOp, ast.USub, ast.UAdd, ast.IfExp,
               ast.And, ast.Or, ast.BoolOp)
    builtins = {'len': len, 'range': range, 'min': min, 'str': str, 'int': int}
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f'Fixture contains unsupported node {type(node).__name__}')
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in builtins):
            raise ValueError('Fixture contains unsupported function call')
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('Fixture contains a reserved name')
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise ValueError('Fixture contains decorators')
    namespace = {'__builtins__': builtins}
    exec(compile(tree, '<generated-test-fixture>', 'exec'), namespace)
    print(json.dumps([namespace[payload['name']](*args) for args in payload['cases']]))


DISTANCE = '''def min_edit_distance(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1
    return dp[n][m]
'''


class LiveEditingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not get('/health')['ready']:
            raise RuntimeError('AXE model is not ready')

    def setUp(self):
        self.wait_idle()

    def wait_idle(self):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not get('/health')['busy']:
                return
            time.sleep(.1)
        self.fail('Generation lock was not released within 30 seconds')

    def revise(self, payload):
        with post(payload) as response:
            self.assertEqual(response.headers.get_content_type(), 'text/event-stream')
            received = list(events(response))
        self.assertTrue(received)
        terminal = [item for item in received if item['type'] in ('done', 'error')]
        self.assertEqual(len(terminal), 1, received[-3:])
        self.assertEqual(terminal[0]['type'], 'done', terminal)
        self.assertEqual(received[-1]['type'], 'done')
        self.assertTrue(any(item['type'] == 'draft' for item in received))
        if payload.get('mode') == 'selection':
            prefix = payload['code'][:payload['selection_start']]
            suffix = payload['code'][payload['selection_end']:]
            for item in received:
                if 'code' in item:
                    self.assertTrue(item['code'].startswith(prefix), item)
                    self.assertTrue(item['code'].endswith(suffix), item)
                    self.assertGreaterEqual(len(item['code']), len(prefix) + len(suffix))
        self.assertGreater(terminal[0]['tokens'], 0)
        return terminal[0]['code']

    def test_selected_bug_fix_preserves_unicode_crlf_and_outside_code(self):
        source = '# café 𝄞 — keep this exactly\r\ndef add(a, b):\r\n    return a - b\r\n\r\nSENTINEL = "<mask>"\r\n'
        start = source.index('a - b')
        revised = self.revise({'code': source, 'instruction': 'Fix the expression so add returns the sum of a and b.',
                               'mode': 'selection', 'selection_start': start, 'selection_end': start+5,
                               'language': 'python', 'max_tokens': 512})
        self.assertEqual(evaluate_fixture(revised, 'add', [[2, 3], [-5, 2], [0, 0]]), [5, -3, 0])

    def test_cursor_insertion_preserves_indentation_and_trailing_newline(self):
        source = 'def square(x):\n    return x * x\n'
        start = source.index('    return')
        revised = self.revise({'code': source, 'instruction': 'Insert an indented one-line docstring documenting that this function returns the square of x. End the inserted line with a newline.',
                               'mode': 'selection', 'selection_start': start, 'selection_end': start,
                               'language': 'python', 'max_tokens': 512})
        self.assertTrue(ast.get_docstring(ast.parse(revised).body[0]))
        self.assertEqual(evaluate_fixture(revised, 'square', [[-3], [0], [7]]), [9, 0, 49])

    def test_selected_minimum_edit_distance_fix(self):
        selected = 'min(dp[i - 1][j], dp[i][j - 1]) + 1'
        start = DISTANCE.index(selected)
        revised = self.revise({'code': DISTANCE, 'instruction': 'Fix the bug in the Levenshtein edit distance calculation.',
                               'mode': 'selection', 'selection_start': start, 'selection_end': start+len(selected),
                               'language': 'python', 'max_tokens': 1024})
        cases = [['', ''], ['', 'abc'], ['abc', ''], ['cat', 'cut'], ['kitten', 'sitting'], ['same', 'same']]
        self.assertEqual(evaluate_fixture(revised, 'min_edit_distance', cases), [0, 3, 3, 1, 3, 0])

    def test_empty_document_insertion(self):
        revised = self.revise({'code': '', 'instruction': 'Insert a Python function square(x) returning x * x.',
                               'mode': 'selection', 'selection_start': 0, 'selection_end': 0,
                               'language': 'python', 'max_tokens': 512})
        self.assertEqual(evaluate_fixture(revised, 'square', [[-3], [0], [7]]), [9, 0, 49])

    def test_selection_deletion_can_empty_document(self):
        source = '# delete this comment\n'
        revised = self.revise({'code': source, 'instruction': 'Delete the selected comment entirely. The replacement must be empty.',
                               'mode': 'selection', 'selection_start': 0, 'selection_end': len(source),
                               'language': 'python', 'max_tokens': 512})
        self.assertEqual(revised, '')

    def test_whole_file_bug_fix(self):
        source = 'def multiply(a, b):\n    return a + b\n'
        revised = self.revise({'code': source, 'instruction': 'Fix the bug. Preserve the function name and arguments.',
                               'mode': 'whole', 'language': 'python', 'max_tokens': 512})
        self.assertEqual(evaluate_fixture(revised, 'multiply', [[2, 3], [-5, 2], [7, 0]]), [6, -10, 0])

    def test_whole_file_documentation_preserves_behavior(self):
        source = 'def square(x):\n    return x * x\n'
        revised = self.revise({'code': source, 'instruction': 'Add a clear docstring and a helpful comment without changing the function name or behavior.',
                               'mode': 'whole', 'language': 'python', 'max_tokens': 1024})
        self.assertTrue(ast.get_docstring(ast.parse(revised).body[0]))
        self.assertEqual(evaluate_fixture(revised, 'square', [[-3], [0], [7]]), [9, 0, 49])

    def test_token_exhaustion_is_an_error_not_a_commit(self):
        with post({'code': 'def square(x):\n    return x * x\n', 'instruction': 'Add detailed documentation.',
                   'mode': 'whole', 'max_tokens': 1}) as response:
            received = list(events(response))
        self.assertEqual(received[-1]['type'], 'error', received)
        self.assertFalse(any(item['type'] == 'done' for item in received))
        self.wait_idle()

    def test_invalid_requests_fail_before_generation(self):
        base = {'code': 'x = 1\n', 'instruction': 'Revise.'}
        cases = [dict(base, mode='selection'), dict(base, mode='selection', selection_start=0, selection_end=99),
                 dict(base, mode='selection', selection_start=4, selection_end=2),
                 dict(base, mode='selection', selection_start='0', selection_end=1),
                 dict(base, mode='whole', selection_start=0, selection_end=1),
                 dict(base, code=' '), dict(base, instruction=' '), dict(base, max_tokens=0)]
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(urllib.error.HTTPError) as error:
                post(payload)
            self.assertIn(error.exception.code, (400, 422))
            error.exception.close()
        self.assertFalse(get('/health')['busy'])

    def test_busy_response_and_disconnect_release_generation_lock(self):
        payload = {'code': 'VALUE = 1\n', 'instruction': 'Add 400 separate, numbered comment lines explaining programming concepts, each on its own line, before the assignment. Do not abbreviate.',
                   'mode': 'whole', 'max_tokens': 8192}
        response = post(payload)
        try:
            first = next(events(response))
            self.assertIn(first['type'], ('draft', 'commit'))
            self.assertTrue(get('/health')['busy'])
            with self.assertRaises(urllib.error.HTTPError) as error:
                post({'code': 'x = 1\n', 'instruction': 'Add a comment.', 'mode': 'whole'})
            self.assertEqual(error.exception.code, 409)
            error.exception.close()
        finally:
            response.close()
        self.wait_idle()
        revised = self.revise({'code': 'def add(a, b):\n    return a - b\n',
                               'instruction': 'Fix addition and keep the function name.', 'mode': 'whole', 'max_tokens': 512})
        self.assertEqual(evaluate_fixture(revised, 'add', [[2, 3]]), [5])


if __name__ == '__main__':
    if '--evaluate' in sys.argv:
        evaluator()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--base', default=BASE)
        args, remaining = parser.parse_known_args()
        BASE = args.base.rstrip('/')
        unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
