# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Exact joint-law and column-binding checks, without a simulator dependency."""

import math
import random
import unittest

import cudaq_edem as edem


def make_model(responses, faults, detectors=(4,), observables=(), successes=()):
    blocks, column_start = [], 0
    for port, fault in enumerate(faults):
        row_start = 0
        for kind, sizes in ((edem.PortKind.DETECTOR, detectors),
                            (edem.PortKind.OBSERVABLE,
                             observables), (edem.PortKind.SUCCESS, successes)):
            for index, size in enumerate(sizes):
                entries = [
                    (row, col)
                    for col in range(fault.size)
                    for row in range(size)
                    if responses[column_start + col] >> (row_start + row) & 1
                ]
                blocks.append(
                    edem.Block(edem.Endpoint(kind, index),
                               edem.Endpoint(edem.PortKind.FAULT, port),
                               edem.SparseMatrix(size, fault.size, entries)))
                row_start += size
        column_start += fault.size
    return edem.project_DEM(
        edem.EDEM(faults=faults,
                  detectors=list(detectors),
                  observables=list(observables),
                  successes=list(successes),
                  blocks=blocks))


def channel(responses, probability):
    count = len(responses)
    return make_model(responses, [
        edem.FaultDistribution(count, [
            edem.FaultGroup(edem.GroupKind.EXCLUSIVE, list(range(count)),
                            [probability / count] * count)
        ])
    ])


def columns(model):
    result, offsets, start = [], [], 0
    for fault in model.faults:
        offsets.append(start)
        result.extend([0] * fault.size)
        start += fault.size
    rows, start = {}, 0
    for kind, sizes in ((edem.PortKind.DETECTOR, model.detectors),
                        (edem.PortKind.OBSERVABLE, model.observables),
                        (edem.PortKind.SUCCESS, model.successes)):
        for index, size in enumerate(sizes):
            rows[kind, index] = start
            start += size
    for block in model.blocks:
        for row, col in block.matrix.entries:
            result[offsets[block.input.index] +
                   col] ^= 1 << (rows[block.output.kind, block.output.index] +
                                 row)
    return result


def joint_law(model):
    responses = columns(model)
    law, start = {0: 1.0}, 0

    def convolve(alternatives):
        nonlocal law
        next_law = {}
        for a, p in law.items():
            for b, q in alternatives:
                next_law[a ^ b] = next_law.get(a ^ b, 0.0) + p * q
        law = next_law

    for fault in model.faults:
        for group in fault.groups:
            terms = [(responses[start + i], p)
                     for i, p in zip(group.fault_ids, group.probabilities)]
            if group.kind == edem.GroupKind.EXCLUSIVE:
                convolve([(0, 1 - math.fsum(group.probabilities)), *terms])
            else:
                for response, p in terms:
                    convolve([(0, 1 - p), (response, p)])
        start += fault.size
    return law


class NoiseTest(unittest.TestCase):

    def assert_same_law(self, left, right):
        a, b = joint_law(left), joint_law(right)
        for outcome in a.keys() | b.keys():
            self.assertAlmostEqual(a.get(outcome, 0),
                                   b.get(outcome, 0),
                                   delta=2e-14)

    def test_exact_pauli_factorization_including_response_collisions(self):
        rng = random.Random(617)
        for width in (2, 4):
            count = (1 << width) - 1
            for probability in (0.0, 0.001, 0.3, count / (count + 1)):
                for trial in range(8):
                    basis = [rng.randrange(16) for _ in range(width)]
                    responses = []
                    for word in range(1, count + 1):
                        response = 0
                        for i, mask in enumerate(basis):
                            if word >> i & 1:
                                response ^= mask
                        responses.append(response)
                    with self.subTest(width=width,
                                      probability=probability,
                                      trial=trial):
                        source = channel(responses, probability)
                        original = source.to_bytes()
                        converted = edem.factorize_dem(source)
                        self.assertEqual(columns(converted), responses)
                        self.assertEqual(converted.detectors, source.detectors)
                        self.assertTrue(
                            all(g.kind == edem.GroupKind.INDEPENDENT
                                for f in converted.faults
                                for g in f.groups))
                        self.assert_same_law(source, converted)
                        reduced, mapping = edem.coalesce_dem_columns(converted)
                        self.assert_same_law(source, reduced)
                        self.assertEqual(len(mapping), count)
                        for i, result in enumerate(mapping):
                            if result is not None:
                                self.assertEqual(
                                    columns(reduced)[result], responses[i])
                        self.assertEqual(source.to_bytes(), original)

    def test_unsupported_laws_are_explicit_failures(self):
        for source in (channel([1, 2, 4], .1), channel([1, 2, 3], .9),
                       channel(list(range(1, 16)), .99),
                       make_model([1, 2, 3], [
                           edem.FaultDistribution(3, [
                               edem.FaultGroup(edem.GroupKind.EXCLUSIVE,
                                               [0, 1, 2], [.1, .2, .1])
                           ])
                       ])):
            with self.subTest(columns=columns(source)):
                with self.assertRaisesRegex(ValueError,
                                            "unsupported fault distribution"):
                    edem.factorize_dem(source)
                with self.assertRaisesRegex(ValueError,
                                            "unsupported fault distribution"):
                    edem.coalesce_dem_columns(source)

    def test_coalescing_respects_all_rows_and_maps_across_partitions(self):
        source = make_model([0, 1, 1, 5, 1, 8, 1], [
            edem.FaultDistribution.independent([.9, .1, .2]),
            edem.FaultDistribution.independent([.3, .4, .5, 0])
        ],
                            detectors=(1, 1),
                            observables=(1,),
                            successes=(1,))
        original = source.to_bytes()
        independent = edem.factorize_dem(source)
        self.assertEqual(independent.to_bytes(), original)
        result, mapping = edem.coalesce_dem_columns(independent)
        self.assertEqual(columns(result), [1, 5, 8])
        self.assertEqual(mapping, [None, 0, 0, 1, 0, 2, 0])
        self.assertEqual(result.detectors, [2])
        self.assertEqual(result.observables, [1])
        self.assertEqual(result.successes, [1])
        self.assert_same_law(source, result)
        again, identity = edem.coalesce_dem_columns(result)
        self.assertEqual(again.to_bytes(), result.to_bytes())
        self.assertEqual(identity, [0, 1, 2])
        self.assertEqual(source.to_bytes(), original)
        restored = edem.DEM.from_bytes(result.to_bytes())
        del source, independent, result
        self.assertEqual(columns(restored), [1, 5, 8])

    def test_stable_parity_probabilities_and_deterministic_cancellation(self):
        for probabilities, expected in (([1, 1], 0), ([1, .1], .9),
                                        ([.5, .2], .5), ([1e-30,
                                                          2e-30], 3e-30)):
            source = make_model(
                [1] * len(probabilities),
                [edem.FaultDistribution.independent(probabilities)])
            result, mapping = edem.coalesce_dem_columns(source)
            if expected == 0:
                self.assertEqual(result.faults, [])
                self.assertEqual(mapping, [None] * len(probabilities))
            else:
                actual = result.faults[0].groups[0].probabilities[0]
                self.assertTrue(
                    math.isclose(actual, expected, rel_tol=1e-14, abs_tol=0))

    def test_empty_values_zero_columns_and_type_safety(self):
        source = edem.project_DEM(edem.EDEM())
        converted = edem.factorize_dem(source)
        result, mapping = edem.coalesce_dem_columns(converted)
        self.assertEqual(result.to_bytes(), source.to_bytes())
        self.assertEqual(mapping, [])
        zero = channel([0, 0, 0], .3)
        reduced, mapping = edem.coalesce_dem_columns(edem.factorize_dem(zero))
        self.assertEqual(mapping, [None, None, None])
        self.assertEqual(reduced.faults, [])
        for operation in (edem.factorize_dem, edem.coalesce_dem_columns):
            with self.assertRaises(TypeError):
                operation(edem.EDEM())

    def test_coalescing_probability_order_is_independent_of_source_order(self):
        probabilities = [1e-30, .1, .2, 1e-16, .99, .001, .3]
        first = make_model([1] * len(probabilities),
                           [edem.FaultDistribution.independent(probabilities)])
        second = make_model(
            [1] * len(probabilities),
            [edem.FaultDistribution.independent(list(reversed(probabilities)))])
        left, _ = edem.coalesce_dem_columns(first)
        right, _ = edem.coalesce_dem_columns(second)
        self.assertEqual(left.to_bytes(), right.to_bytes())


if __name__ == "__main__":
    unittest.main()
