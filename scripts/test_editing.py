#!/usr/bin/env python3
"""Unit regressions using the real checkpoint tokenizer, with no model load or mocks.

Run in the serving image with the test directory mounted:
  docker compose run --rm --no-deps -v "$PWD/scripts:/app/scripts:ro" xe-diffusion -m unittest scripts.test_editing -v
"""
from pathlib import Path
import queue
import sys
import threading
import unittest

import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import CancelledGeneration, EditRequest, EditStreamer, REPO, REVISION, edit_replacement, edit_source


class FramingTests(unittest.TestCase):
    def test_source_whitespace_and_literal_special_tokens(self):
        source = '    value = "<mask> <pad> <|channel>thought\\n"\r\n\r\n'
        self.assertEqual(edit_source(source, final=True), source)

    def test_only_leading_reasoning_channel_is_removed(self):
        source = '    return "<channel|>"\n'
        self.assertEqual(edit_source('<|channel>thought\nprivate\n<channel|>' + source, final=True), source)
        self.assertEqual(edit_source('<|channel>thought\nstill thinking'), '')
        with self.assertRaises(ValueError):
            edit_source('<|channel>thought\nstill thinking', final=True)

    def test_fences_preserve_source_indentation_and_newlines(self):
        source = '    return 3\n\n'
        self.assertEqual(edit_source('\n```python\n' + source + '```\n\n', final=True), source)
        with self.assertRaises(ValueError):
            edit_source('```python\nreturn 3', final=True)

    def test_replacement_preserves_boundary_whitespace(self):
        for replacement in ('', 'a + b', '    """Doc."""\n', '\r\n\treturn 1\r\n'):
            with self.subTest(replacement=replacement):
                self.assertEqual(edit_replacement('<replacement>' + replacement + '</replacement>', final=True), replacement)

    def test_replacement_can_contain_literal_closing_marker(self):
        replacement = 'token = "</replacement>"\n'
        self.assertEqual(edit_replacement('<replacement>' + replacement + '</replacement>', final=True), replacement)

    def test_partial_and_invalid_boundaries(self):
        self.assertIsNone(edit_replacement('<repla'))
        self.assertEqual(edit_replacement('<replacement>a + b</repla'), 'a + b')
        for raw in ('a + b', '<replacement>a + b', '<replacement>x</replacement>explanation'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                edit_replacement(raw, final=True)


class StreamerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = snapshot_download(REPO, revision=REVISION, local_files_only=True,
                                 allow_patterns=['*.safetensors', '*.json', '*.jinja'])
        cls.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        cls.eos = cls.tokenizer.eos_token_id

    def streamer(self, request, raw, *, eos=True, budget=4096):
        events = queue.Queue()
        stream = EditStreamer(self.tokenizer, {self.eos}, events, threading.Event(), request, budget)
        stream.put(torch.tensor([[self.tokenizer.bos_token_id]]))
        ids = self.tokenizer.encode(raw, add_special_tokens=False)
        stream.put(torch.tensor([ids + ([self.eos] if eos else [])], dtype=torch.long))
        return stream, events

    def test_real_tokenizer_preserves_literal_special_tokens(self):
        source = 'value = "<mask> <pad>"\n'
        request = EditRequest(code=source, instruction='Keep source.', mode='whole')
        stream, _ = self.streamer(request, '<|channel>thought\n<channel|><replacement>' + source + '</replacement>')
        self.assertEqual(stream.final_source(), source)

    def test_unicode_crlf_selection_is_exact(self):
        source = '# café 𝄞\r\ndef add(a, b):\r\n    return a - b\r\n# unchanged\r\n'
        start = source.index('a - b')
        request = EditRequest(code=source, instruction='Fix addition.', mode='selection',
                              selection_start=start, selection_end=start+5)
        stream, events = self.streamer(request, '<replacement>a + b</replacement>')
        expected = source[:start] + 'a + b' + source[start+5:]
        self.assertEqual(stream.final_source(), expected)
        self.assertEqual(events.get_nowait()['code'], expected)

    def test_cursor_insertion_and_empty_replacement_deletion(self):
        source = 'def square(x):\n    return x * x\n'
        start = source.index('    return')
        request = EditRequest(code=source, instruction='Document it.', mode='selection',
                              selection_start=start, selection_end=start)
        replacement = '    """Return the square."""\n'
        stream, _ = self.streamer(request, '<replacement>' + replacement + '</replacement>')
        self.assertEqual(stream.final_source(), source[:start] + replacement + source[start:])
        source = '# remove me\nx = 1\n'
        request = EditRequest(code=source, instruction='Delete comment.', mode='selection',
                              selection_start=0, selection_end=source.index('x'))
        stream, _ = self.streamer(request, '<replacement></replacement>')
        self.assertEqual(stream.final_source(), 'x = 1\n')

    def test_empty_document_insertion_and_full_deletion(self):
        request = EditRequest(code='', instruction='Insert assignment.', mode='selection',
                              selection_start=0, selection_end=0)
        stream, _ = self.streamer(request, '<replacement>x = 1\n</replacement>')
        self.assertEqual(stream.final_source(), 'x = 1\n')
        request = EditRequest(code='x = 1\n', instruction='Delete all.', mode='selection',
                              selection_start=0, selection_end=6)
        stream, _ = self.streamer(request, '<replacement></replacement>')
        self.assertEqual(stream.final_source(), '')

    def test_eos_budget_and_syntax_are_required(self):
        request = EditRequest(code='x = 1\n', instruction='Revise.', mode='whole')
        for raw, eos, budget in [('x = 2\n', False, 100), ('x = 2\n', True, 1), ('def broken(', True, 100), ('', True, 100)]:
            with self.subTest(raw=raw, eos=eos, budget=budget):
                stream, _ = self.streamer(request, '<replacement>' + raw + '</replacement>', eos=eos, budget=budget)
                with self.assertRaises(ValueError):
                    stream.final_source()

    def test_python_is_compiled_not_only_parsed(self):
        request = EditRequest(code='x = 1\n', instruction='Revise.', mode='whole')
        for invalid in ('return 1\n', 'break\n', 'nonlocal missing\n'):
            with self.subTest(invalid=invalid):
                stream, _ = self.streamer(request, '<replacement>' + invalid + '</replacement>')
                with self.assertRaisesRegex(ValueError, 'not valid Python'):
                    stream.final_source()

    def test_whole_file_preserves_docstrings_and_literal_escapes(self):
        source = '\"\"\"Module documentation.\"\"\"\npattern = r"\\n"\n'
        request = EditRequest(code=source, instruction='Keep source.', mode='whole')
        stream, _ = self.streamer(request, '<replacement>' + source + '</replacement>')
        self.assertEqual(stream.final_source(), source)
        broken = source.replace('\"\"\"\n', '\\\"\\\"\\\"\n')
        stream, _ = self.streamer(request, '<replacement>' + broken + '</replacement>')
        with self.assertRaises(ValueError):
            stream.final_source()

    def test_final_validation_checks_assembled_document(self):
        source = 'def answer():\n    return 41\n'
        start = source.index('41')
        request = EditRequest(code=source, instruction='Revise.', mode='selection',
                              selection_start=start, selection_end=start+2)
        stream, _ = self.streamer(request, '<replacement>42</replacement>')
        self.assertEqual(stream.final_source(), source.replace('41', '42'))
        stream, _ = self.streamer(request, '<replacement>) invalid</replacement>')
        with self.assertRaises(ValueError):
            stream.final_source()

    def test_cancelled_callback_does_not_publish_a_draft(self):
        request = EditRequest(code='x = 1\n', instruction='Revise.')
        events = queue.Queue()
        cancelled = threading.Event()
        stream = EditStreamer(self.tokenizer, {self.eos}, events, cancelled, request, 100)
        cancelled.set()
        with self.assertRaises(CancelledGeneration):
            stream.put_draft(torch.tensor([[1, 2, 3]]))
        self.assertTrue(events.empty())


if __name__ == '__main__':
    unittest.main()
