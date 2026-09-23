#!/usr/bin/env python3
"""Actual compiler/sandbox tests; no mocked processes or model responses."""
import json
from pathlib import Path
import tempfile
import time
import unittest

from scripts import edit_bench as bench


class CheckerTests(unittest.TestCase):
    def test_public_examples_exclude_reference_answers(self):
        examples = bench.public_cases()
        self.assertTrue(examples)
        for case in examples:
            self.assertNotIn('reference', case)
            self.assertNotIn('harness', case)
            self.assertIn('dataset_task', case)
            self.assertEqual(case['language'], 'python')
            self.assertIn('starter_code', case)
            self.assertGreater(case['selection_end'], case['selection_start'])
            self.assertLessEqual(case['selection_end'], len(case['code']))
            self.assertNotEqual(case['prompts']['vague'], case['prompts']['detailed'])

    def test_rejects_unknown_case_and_oversized_source(self):
        with self.assertRaises(ValueError):
            bench.check_code('../server.py', 'int main() {}')
        with self.assertRaises(ValueError):
            bench.check_code('two-sum', 'x' * 150001)
        with self.assertRaises(ValueError):
            bench.check_code([], '')

    def test_real_compiler_error_is_not_a_test_pass(self):
        result = bench.check_code('intervals', 'this is not C++;')
        self.assertEqual(result['status'], 'compile_error')
        self.assertNotEqual(result['compile']['returncode'], 0)
        self.assertNotIn('run', result)

    def test_independent_harness_rejects_bug_and_accepts_control(self):
        case = bench.MANIFEST['intervals']
        broken = bench.check_code('intervals', (bench.CASES / case['file']).read_text())
        correct = bench.check_code('intervals', case['reference'])
        self.assertEqual(broken['status'], 'test_failure')
        self.assertIn('nested interval', broken['run']['output'])
        self.assertEqual(correct['status'], 'pass')
        self.assertIn('interval checks', correct['run']['output'])

    def test_runtime_filesystem_network_and_resource_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / 'probe.py').write_text('''from pathlib import Path
import resource
import socket
assert not Path('/home').exists()
assert not Path('/root').exists()
for target in ['/usr/axe-forbidden-write', '/work/probe.py']:
    try:
        Path(target).write_text('not allowed')
    except OSError:
        pass
    else:
        raise AssertionError('read-only mount was writable')
try:
    socket.create_connection(('127.0.0.1', 8080), timeout=.2)
except OSError:
    pass
else:
    raise AssertionError('host network reachable')
assert resource.getrlimit(resource.RLIMIT_AS)[0] < 2 * 1024**3
assert resource.getrlimit(resource.RLIMIT_CPU)[0] != resource.RLIM_INFINITY
Path('/tmp/allowed').write_text('scratch only')
print('ISOLATION_OK')
''')
            result = bench.execute(work, ['/usr/bin/python3', '-I', '/work/probe.py'])
        self.assertEqual(result['returncode'], 0, result)
        self.assertIn('ISOLATION_OK', result['output'])

    def test_zero_exit_without_tests_is_not_a_pass(self):
        result = bench.check_code('two-sum', 'raise SystemExit(0)\n')
        self.assertEqual(result['run']['returncode'], 0)
        self.assertEqual(result['status'], 'invalid_test_result')
        source = '#include <cstdlib>\nstatic int premature = (std::exit(0), 0);\n' + bench.MANIFEST['intervals']['reference']
        result = bench.check_code('intervals', source)
        self.assertEqual(result['run']['returncode'], 0)
        self.assertEqual(result['status'], 'invalid_test_result')

    def test_hung_process_is_killed_at_wall_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / 'sleep.py').write_text('import time\ntime.sleep(120)\n')
            started = time.monotonic()
            result = bench.execute(work, ['/usr/bin/python3', '-I', '/work/sleep.py'])
            elapsed = time.monotonic() - started
        self.assertTrue(result['timeout'], result)
        self.assertNotEqual(result['returncode'], 0)
        self.assertLess(elapsed, 75)


class LeetCodeCheckerTests(unittest.TestCase):
    def test_python_compile_failure_stops_before_execution(self):
        result = bench.check_code('two-sum', 'def broken(:\n')
        self.assertEqual(result['status'], 'compile_error')
        self.assertNotIn('run', result)

    def test_zero_exit_without_dataset_checks_is_not_a_pass(self):
        result = bench.check_code('two-sum', 'raise SystemExit(0)\n', 'starter')
        self.assertEqual(result['run']['returncode'], 0)
        self.assertEqual(result['status'], 'invalid_test_result')

    def test_starter_uses_the_supplied_dataset_environment(self):
        reference = bench.DATASET_RECORDS['sliding-window-maximum']['reference']
        solution = 'class Solution:' + reference.split('class Solution:', 1)[1]
        standalone = bench.check_code('sliding-window-maximum', solution, 'file')
        with_environment = bench.check_code('sliding-window-maximum', solution, 'starter')
        self.assertEqual(standalone['status'], 'test_failure')
        self.assertEqual(with_environment['status'], 'pass')
        self.assertEqual(with_environment['tests']['passed'], with_environment['tests']['total'])

    def test_unknown_workflow_is_rejected(self):
        with self.assertRaises(ValueError):
            bench.check_code('two-sum', 'pass', 'unknown')


if __name__ == '__main__':
    unittest.main()
