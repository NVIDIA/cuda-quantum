# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                            #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""CPU-only regression tests for the Shor tutorial's classical post-processing.

Run without a CUDA-Q installation using:
  python python/tests/test_shors_tutorial.py
"""

import ast
import contextlib
import io
import json
import math
from pathlib import Path
import random
import unittest
from unittest import mock


def load_classical_functions():
    """Load the notebook's real functions without executing quantum cells."""
    path = (Path(__file__).resolve().parents[2] /
            'docs/sphinx/applications/python/shors.ipynb')
    notebook = json.loads(path.read_text(encoding='utf-8'))
    names = {'test_order', 'find_order_classical', 'shors_algorithm'}
    namespace = {'gcd': math.gcd, 'random': random}
    for cell in notebook['cells']:
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] != 'code' or not any(f'def {name}(' in source
                                                  for name in names):
            continue
        tree = ast.parse(source)
        definitions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), str(path),
                    'exec'), namespace)
    if not names.issubset(namespace):
        raise AssertionError('The tutorial is missing a classical helper')
    return namespace


class ShorPostProcessingTest(unittest.TestCase):

    def setUp(self):
        self.functions = load_classical_functions()
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_factor_extraction_uses_the_half_power(self):
        for a, order, modulus, expected in ((2, 12, 35, (7, 5)), (5, 3936, 8051,
                                                                  (83, 97)),
                                            (2, 4, 15, (3, 5))):
            with self.subTest(a=a, order=order, modulus=modulus):
                result = self.functions['test_order'](a, order, modulus)
                self.assertEqual(result, expected)
                self.assertEqual(math.prod(result), modulus)

    def test_order_failure_always_returns_a_pair(self):
        for a, order, modulus in ((5, 6, 21), (1, 2, 15), (2, None, 15),
                                  (4, 3, 21), (2, 2, 15)):
            with self.subTest(a=a, order=order, modulus=modulus):
                self.assertEqual(
                    self.functions['test_order'](a, order, modulus), (0, 0))

    def test_classical_driver_factors_the_reported_example(self):
        self.assertEqual(self.functions['shors_algorithm'](35, 2, False),
                         (7, 5))

    def test_driver_retries_after_the_negative_one_residue(self):
        # The first base gives order 6, but 5**3 == -1 modulo 21. The next
        # deliberately chosen base has a common factor and needs no quantum call.
        with mock.patch.object(random, 'choice', return_value=3) as choose:
            self.assertEqual(self.functions['shors_algorithm'](21, 5, False),
                             (3, 7))
        choose.assert_called_once()
        self.assertNotIn(5, choose.call_args.args[0])

    def test_bounded_classical_order_sweep(self):
        # These are exact small arithmetic cases, independent of any simulator.
        for modulus in (15, 21, 33, 35, 55, 65, 77, 91):
            for a in range(2, modulus):
                if math.gcd(a, modulus) != 1:
                    continue
                order = self.functions['find_order_classical'](a, modulus)
                result = self.functions['test_order'](a, order, modulus)
                with self.subTest(a=a, order=order, modulus=modulus):
                    self.assertIsInstance(result, tuple)
                    self.assertEqual(len(result), 2)
                    if order % 2 or pow(a, order // 2, modulus) == modulus - 1:
                        self.assertEqual(result, (0, 0))
                    else:
                        self.assertTrue(
                            all(1 < factor < modulus for factor in result))
                        self.assertEqual(math.prod(result), modulus)


if __name__ == '__main__':
    unittest.main()
