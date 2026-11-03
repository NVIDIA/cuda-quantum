# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Value conformance through the public Python producer interface."""

import argparse
import gc
import hashlib
import importlib
import itertools
import random
import struct
import sys
import unittest
import weakref

from cudaq_edem import (
    CodePresentation,
    DEM,
    EDEM,
    Endpoint,
    EndpointDirection,
    FaultDistribution,
    FaultGroup,
    MatrixBlock,
    GroupKind,
    PortKind,
    SparseMatrix,
    compose_chain,
    project_DEM,
)

try:
    _adoption_test = importlib.import_module("cudaq_edem._adoption_test")
except ModuleNotFoundError as error:
    if error.name != "cudaq_edem._adoption_test":
        raise
    # The native test producer is deliberately absent from installed packages.
    _adoption_test = None


class Layout:
    """A little-endian packed record, stated once by its named fields."""

    def __init__(self, **fields):
        self.codes = fields
        self.struct = struct.Struct("<" + "".join(fields.values()))
        self.size = self.struct.size
        self.offsets, prefix = {}, "<"
        for name, code in fields.items():
            self.offsets[name] = struct.calcsize(prefix)
            prefix += code

    def unpack(self, data, offset=0):
        return dict(zip(self.codes, self.struct.unpack_from(data, offset)))

    def pack_into(self, buffer, offset, field, value):
        struct.pack_into("<" + self.codes[field], buffer,
                         offset + self.offsets[field], value)


# These tests pin the exchange format; its records are described only here.
HEADER = Layout(magic="8s",
                version="I",
                kind="I",
                size="Q",
                alignment="I",
                index_bits="I",
                directory_offset="Q",
                directory_count="Q",
                directory_bytes="Q",
                reserved="Q")
SECTION = Layout(id="I", flags="I", count="Q", bytes="Q", offset="Q")
TYPE = Layout(name="Q", name_size="Q", role="I", wire_count="I")
BUNDLE = Layout(type="I", reserved="I", widths="Q", offsets="Q")
PRESENTATION = Layout(type="I",
                      syndrome_count="I",
                      virtual_outcome_count="I",
                      logical_qubit_count="I")
SECTION_OUTPUTS, SECTION_PRESENTATIONS = 3, 0x10000


def directory(packed):
    header = HEADER.unpack(packed)
    return [
        SECTION.unpack(packed, header["directory_offset"] + SECTION.size * i)
        for i in range(header["directory_count"])
    ]


def section(packed, section_id):
    return next(e for e in directory(packed) if e["id"] == section_id)


def open_round():
    """An explicit repetition-code seam; this fixture is not circuit compilation.

    f0 flips the first check measurement. f1 is an outgoing data error after
    measurement, with syndrome (1,0) and zero chosen logical coordinates.
    """
    identity = SparseMatrix(2, 2, [(0, 0), (1, 1)])
    measurement = SparseMatrix(2, 2, [(0, 0)])
    data = SparseMatrix(2, 2, [(0, 1)])
    edges = [
        (PortKind.DETECTOR, PortKind.FAULT, measurement),
        (PortKind.DETECTOR, PortKind.VIRTUAL, identity),
        (PortKind.DETECTOR, PortKind.SYNDROME, identity),
        (PortKind.VIRTUAL, PortKind.FAULT, measurement),
        (PortKind.VIRTUAL, PortKind.SYNDROME, identity),
        (PortKind.SYNDROME, PortKind.FAULT, data),
        (PortKind.SYNDROME, PortKind.SYNDROME, identity),
        (PortKind.LOGICAL, PortKind.LOGICAL, identity),
    ]
    return EDEM(
        codes=[CodePresentation("repetition-3", 2, 2, 1)],
        inputs=[0],
        outputs=[0],
        faults=[FaultDistribution.independent([0.01, 0.02])],
        detectors=[2],
        blocks=[
            MatrixBlock(Endpoint(out), Endpoint(inp), m)
            for out, inp, m in edges
        ],
    )


class CompositionTest(unittest.TestCase):

    def test_repeated_rounds_keep_occurrence_faults_and_exact_laws(self):
        original = open_round()
        source = EDEM(
            codes=original.codes,
            inputs=original.inputs,
            outputs=original.outputs,
            detectors=original.detectors,
            faults=[
                FaultDistribution(
                    2, [FaultGroup(GroupKind.EXCLUSIVE, [0, 1], [0.1, 0.2])])
            ],
            blocks=original.blocks)
        composed = compose_chain([source, source])
        self.assertFalse(composed.is_dem)
        self.assertEqual(composed.inputs, [0])
        self.assertEqual(composed.outputs, [0])
        self.assertEqual(composed.detectors, [2, 2])
        self.assertEqual(len(composed.faults), 2)
        for distribution in composed.faults:
            self.assertEqual(distribution.size, 2)
            self.assertEqual(len(distribution.groups), 1)
            group = distribution.groups[0]
            self.assertEqual(group.kind, GroupKind.EXCLUSIVE)
            self.assertEqual(group.fault_ids, [0, 1])
            self.assertEqual(group.probabilities, [0.1, 0.2])
        for bits in itertools.product((0, 1), repeat=10):
            first = source.apply([*bits[:2], *bits[4:]])
            second = source.apply([*bits[2:4], *first[2:]])
            self.assertEqual(composed.apply(bits), [*first[:2], *second])
        packed = composed.to_bytes()
        del source, original
        gc.collect()
        self.assertEqual(composed.to_bytes(), packed)
        self.assertEqual(EDEM.from_bytes(packed).to_bytes(), packed)

    def test_projection_owns_exact_closed_value_and_roundtrips(self):
        source = EDEM(faults=[
            FaultDistribution(3, [
                FaultGroup(GroupKind.EXCLUSIVE, [0, 2], [0.2, 0.3]),
                FaultGroup(GroupKind.INDEPENDENT, [1], [0.1]),
            ])
        ],
                      detectors=[2],
                      observables=[1],
                      successes=[1],
                      blocks=[
                          MatrixBlock(
                              Endpoint(PortKind.DETECTOR),
                              Endpoint(PortKind.FAULT),
                              SparseMatrix(2, 3, [(0, 0), (1, 1), (0, 2)])),
                          MatrixBlock(Endpoint(PortKind.OBSERVABLE),
                                      Endpoint(PortKind.FAULT),
                                      SparseMatrix(1, 3, [(0, 2)])),
                          MatrixBlock(Endpoint(PortKind.SUCCESS),
                                      Endpoint(PortKind.FAULT),
                                      SparseMatrix(1, 3, [(0, 1)])),
                      ])
        packed = source.to_bytes()
        result = project_DEM(source)
        self.assertIsInstance(result, DEM)
        self.assertNotIsInstance(result, EDEM)
        self.assertTrue(source.is_dem)
        self.assertTrue(result.is_dem)
        self.assertNotEqual(result._data_address, source._data_address)
        expected = [
            source.apply(bits) for bits in itertools.product((0, 1), repeat=3)
        ]
        del source
        gc.collect()
        self.assertEqual(result.to_bytes(), packed)
        self.assertEqual(result.detectors, [2])
        self.assertEqual(result.observables, [1])
        self.assertEqual(result.successes, [1])
        self.assertEqual(result.inputs, [])
        self.assertEqual(result.outputs, [])
        self.assertEqual(len(result.blocks), 3)
        self.assertEqual(result.faults[0].groups[1].probabilities, [0.2, 0.3])
        self.assertEqual(result.port(Endpoint(PortKind.FAULT)), (3, 0))
        for loaded in (result, DEM.from_bytes(packed)):
            self.assertEqual(loaded.to_bytes(), packed)
            self.assertEqual([
                loaded.apply(bits)
                for bits in itertools.product((0, 1), repeat=3)
            ], expected)
        with self.assertRaises(TypeError):
            DEM()

    def test_projection_rejects_all_structural_boundaries(self):
        for syndrome_count, logical_qubit_count in ((0, 0), (1, 1)):
            code = CodePresentation("boundary", syndrome_count, syndrome_count,
                                    logical_qubit_count)
            for side in ("inputs", "outputs"):
                source = EDEM(codes=[code], **{side: [0]})
                self.assertFalse(source.is_dem)
                with self.assertRaises(ValueError):
                    project_DEM(source)
                with self.assertRaises(ValueError):
                    DEM.from_bytes(source.to_bytes())
        empty = project_DEM(EDEM())
        self.assertEqual(empty.vector_sizes, (0, 0))
        self.assertEqual(empty.apply([]), [])

    def test_composition_rejects_invalid_chains(self):
        with self.assertRaises(ValueError):
            compose_chain([])
        with self.assertRaises((TypeError, ValueError)):
            compose_chain([None])
        with self.assertRaises(TypeError):
            compose_chain([object()])
        with self.assertRaises(ValueError):
            compose_chain([open_round(), EDEM()])
        left = EDEM(codes=[CodePresentation("left", 0, 0, 0)], outputs=[0])
        right = EDEM(codes=[CodePresentation("right", 0, 0, 0)], inputs=[0])
        with self.assertRaises(ValueError):
            compose_chain([left, right])
        # Names identify codes across boundaries; counts must also match.
        right = EDEM(codes=[CodePresentation("left", 1, 0, 0)], inputs=[0])
        with self.assertRaises(ValueError):
            compose_chain([left, right])
        right = EDEM(codes=[CodePresentation("left", 0, 0, 0)], inputs=[0])
        self.assertTrue(compose_chain([left, right]).is_dem)
        self.assertEqual(compose_chain([EDEM()]).to_bytes(), EDEM().to_bytes())


class ValuesTest(unittest.TestCase):

    def test_adoption_rejects_non_capsules(self):
        for source in (None, b"", object(), open_round()):
            with self.assertRaises(TypeError):
                EDEM.adopt(source)

    def test_sparse_construction_against_dense_parity(self):
        rng = random.Random(391)
        for rows, columns in itertools.product(range(5), repeat=2):
            entries = ([
                (rng.randrange(rows), rng.randrange(columns)) for _ in range(31)
            ] if rows and columns else [])
            matrix = SparseMatrix(rows, columns, entries)
            for x in itertools.product((0, 1), repeat=columns):
                expected = [0] * rows
                for r, c in entries:
                    expected[r] ^= x[c]
                self.assertEqual(matrix.apply(x), expected)
            self.assertEqual(matrix.shape, (rows, columns))
        self.assertEqual(SparseMatrix(1, 1, [(0, 0)] * 2).entries, [])

    def test_affine_and_invalid_vectors(self):
        matrix = SparseMatrix(2, 2, [(0, 0), (1, 1)])
        self.assertEqual(matrix.apply([1, 0], offset=[1, 1]), [0, 1])
        for values in ([1], [0, 2]):
            with self.assertRaises(ValueError):
                matrix.apply(values)
        with self.assertRaises(ValueError):
            matrix.apply([0, 0], offset=[0])
        with self.assertRaises(ValueError):
            SparseMatrix(2, 2, [(2, 0)])

    def test_open_round_all_assignments(self):
        value = open_round()
        self.assertEqual(value.vector_sizes, (8, 8))
        self.assertEqual(value.port(Endpoint(PortKind.LOGICAL)), (2, 6))
        self.assertEqual(
            value.port(Endpoint(PortKind.VIRTUAL),
                       direction=EndpointDirection.OUTPUT), (2, 2))
        for f0, f1, v0, v1, s0, s1, l0, l1 in itertools.product((0, 1),
                                                                repeat=8):
            self.assertEqual(
                value.apply([f0, f1, v0, v1, s0, s1, l0, l1]),
                [v0 ^ s0 ^ f0, v1 ^ s1, s0 ^ f0, s1, s0 ^ f1, s1, l0, l1],
            )
        # Late fault survives in S_out but does not alter this round's records.
        self.assertEqual(value.apply([0, 1, 0, 0, 0, 0, 0, 0]),
                         [0, 0, 0, 0, 1, 0, 0, 0])

    def test_roundtrip_and_owner_lifetimes(self):
        original = open_round()
        packed = original.to_bytes()
        # Golden bytes for the typed-map format with a trailing directory.
        self.assertEqual(
            hashlib.sha256(packed).hexdigest(),
            "4335c484e15e0b4270886909e83f05556284dfb649e2ff579bc611b4cef513cf")
        loaded = EDEM.from_bytes(packed)
        blocks = loaded.blocks
        del original, loaded
        gc.collect()
        self.assertEqual(blocks[0].matrix.apply([1, 0]), [1, 0])
        loaded = EDEM.from_bytes(packed)
        self.assertEqual(loaded.to_bytes(), packed)
        self.assertEqual(loaded.codes[0].name, "repetition-3")
        self.assertEqual(loaded.inputs, [0])
        self.assertEqual(loaded.outputs, [0])
        self.assertEqual(loaded.detectors, [2])
        self.assertEqual(loaded.observables, [])
        self.assertEqual(loaded.successes, [])
        self.assertEqual(loaded.faults[0].groups[0].probabilities, [0.01, 0.02])
        self.assertEqual(len(loaded.blocks), 8)
        with self.assertRaises(AttributeError):
            loaded.inputs = []

    def test_typed_layout_and_presentation_checks(self):
        packed = open_round().to_bytes()
        header = HEADER.unpack(packed)
        self.assertEqual(header["magic"], b"CQGF2\0\0\0")
        self.assertEqual(header["size"], len(packed))
        self.assertEqual((header["alignment"], header["index_bits"]), (8, 32))
        self.assertEqual(header["directory_bytes"],
                         SECTION.size * header["directory_count"])
        self.assertEqual(header["directory_offset"] + header["directory_bytes"],
                         len(packed))
        entries = directory(packed)
        self.assertEqual([e["id"] for e in entries],
                         [1, 2, 3, 4, 5, 0x10000, 0x10001])
        types = entries[0]
        names = []
        for i in range(types["count"]):
            t = TYPE.unpack(packed, types["offset"] + TYPE.size * i)
            start = types["offset"] + t["name"]
            names.append(packed[start:start + t["name_size"]].decode())
            self.assertEqual((t["role"], t["wire_count"]), (1, 3) if i == 0 else
                             (0, 1))
        self.assertEqual(names, [
            "edem.code/repetition-3", "edem.detector", "edem.fault",
            "edem.observable", "edem.success"
        ])
        codes = section(packed, SECTION_PRESENTATIONS)
        self.assertEqual(
            PRESENTATION.struct.unpack_from(packed, codes["offset"]),
            (0, 2, 2, 1))
        # Widths do not replace the retained presentation metadata check.
        broken = bytearray(packed)
        PRESENTATION.pack_into(broken, codes["offset"], "syndrome_count", 3)
        with self.assertRaises(ValueError):
            EDEM.from_bytes(bytes(broken))

    def test_canonical_construction(self):
        value = open_round()
        zero = MatrixBlock(Endpoint(PortKind.DETECTOR),
                           Endpoint(PortKind.LOGICAL), SparseMatrix(2, 2))
        rebuilt = EDEM(codes=value.codes,
                       inputs=value.inputs,
                       outputs=value.outputs,
                       faults=value.faults,
                       detectors=[2],
                       blocks=[zero, *reversed(value.blocks)])
        self.assertEqual(rebuilt.to_bytes(), value.to_bytes())
        # Presentation order is local; bytes depend on the named presentation.
        a = CodePresentation("a", 0, 0, 1)
        z = CodePresentation("z", 0, 0, 1)
        x = EDEM(codes=[z, a], inputs=[0], outputs=[1])
        y = EDEM(codes=[a, z], inputs=[1], outputs=[0])
        self.assertEqual(x.to_bytes(), y.to_bytes())
        self.assertEqual([c.name for c in x.codes], ["a", "z"])
        # Names are unique within a code table.
        with self.assertRaises(ValueError):
            EDEM(codes=[a, CodePresentation("a", 0, 0, 1)])
        packed = x.to_bytes()
        codes = section(packed, SECTION_PRESENTATIONS)
        # Each presentation record names its own type ID.
        broken = bytearray(packed)
        PRESENTATION.pack_into(broken, codes["offset"] + PRESENTATION.size,
                               "type", 0)
        with self.assertRaises(ValueError):
            EDEM.from_bytes(bytes(broken))
        # Packed EDEMs retain only presentations used by code bundles: point
        # the output bundle at "z" so "a" is left unused.
        outputs = section(packed, SECTION_OUTPUTS)
        broken = bytearray(packed)
        BUNDLE.pack_into(broken, outputs["offset"], "type", 1)
        with self.assertRaises(ValueError):
            EDEM.from_bytes(bytes(broken))

    def test_retains_used_presentations(self):
        codes = [CodePresentation(f"code-{i}", 0, 0, 1) for i in range(50)]
        first = EDEM(codes=codes, inputs=[7], outputs=[3])
        second = EDEM(codes=codes, inputs=[3], outputs=[7])
        # Each EDEM keeps only the presentations it uses, in name order.
        self.assertEqual([c.name for c in first.codes], ["code-3", "code-7"])
        self.assertEqual(first.inputs, [1])
        self.assertEqual(first.outputs, [0])
        composed = compose_chain([first, second])
        self.assertEqual([c.name for c in composed.codes], ["code-7"])
        self.assertEqual((composed.inputs, composed.outputs), ([0], [0]))
        self.assertEqual(
            EDEM.from_bytes(first.to_bytes()).to_bytes(), first.to_bytes())
        with self.assertRaises(ValueError):
            EDEM(inputs=[0])
        with self.assertRaises(ValueError):
            EDEM(codes=codes[:1], inputs=[1])
        # The whole table is checked, including unused presentations.
        with self.assertRaises(ValueError):
            EDEM(codes=[CodePresentation("", 0, 0, 0)])

    def test_many_distinct_and_shared_matrices(self):
        blocks = [
            MatrixBlock(Endpoint(PortKind.DETECTOR, i),
                        Endpoint(PortKind.FAULT),
                        SparseMatrix(64, 64, [(i % 32, (i // 32) % 16)]))
            for i in range(600)
        ]
        arguments = dict(faults=[FaultDistribution.independent([0.001] * 64)],
                         detectors=[64] * len(blocks))
        ordered = EDEM(**arguments, blocks=blocks)
        random.Random(93).shuffle(blocks)
        shuffled = EDEM(**arguments, blocks=blocks)
        self.assertEqual(ordered.to_bytes(), shuffled.to_bytes())
        self.assertEqual(
            EDEM.from_bytes(ordered.to_bytes()).to_bytes(), ordered.to_bytes())
        x = [i % 2 for i in range(64)]
        expected = [0] * (64 * len(blocks))
        for i in range(len(blocks)):
            expected[64 * i + i % 32] = x[(i // 32) % 16]
        self.assertEqual(ordered.apply(x), expected)

    def test_invalid_boundaries_and_structural_zeros(self):
        code = CodePresentation("test", 1, 1, 1)
        for output in (PortKind.SYNDROME, PortKind.LOGICAL):
            rows = 2 if output == PortKind.LOGICAL else 1
            with self.assertRaises(ValueError):
                EDEM(codes=[code],
                     inputs=[0],
                     outputs=[0],
                     observables=[1],
                     blocks=[
                         MatrixBlock(Endpoint(output),
                                     Endpoint(PortKind.VIRTUAL),
                                     SparseMatrix(rows, 1, [(0, 0)]))
                     ])
        observed = EDEM(codes=[code],
                        inputs=[0],
                        outputs=[0],
                        observables=[1],
                        blocks=[
                            MatrixBlock(Endpoint(PortKind.OBSERVABLE),
                                        Endpoint(PortKind.VIRTUAL),
                                        SparseMatrix(1, 1, [(0, 0)]))
                        ])
        packed = observed.to_bytes()
        self.assertEqual(EDEM.from_bytes(packed).to_bytes(), packed)
        with self.assertRaises(ValueError):
            EDEM(codes=[code], inputs=[1])
        with self.assertRaises(ValueError):
            EDEM(codes=[code],
                 inputs=[0],
                 detectors=[2],
                 blocks=[
                     MatrixBlock(Endpoint(PortKind.DETECTOR),
                                 Endpoint(PortKind.SYNDROME),
                                 SparseMatrix(1, 1))
                 ])
        duplicate = open_round().blocks[0]
        with self.assertRaises(ValueError):
            EDEM(codes=open_round().codes,
                 inputs=[0],
                 outputs=[0],
                 detectors=[2],
                 faults=open_round().faults,
                 blocks=[duplicate, duplicate])

    def test_exclusive_laws_and_group_order(self):
        normalized = [1.0 / 9.0] * 9
        rounded = EDEM(faults=[
            FaultDistribution(
                9,
                [FaultGroup(GroupKind.EXCLUSIVE, list(range(9)), normalized)])
        ])
        self.assertEqual(rounded.faults[0].groups[0].probabilities, normalized)
        self.assertEqual(
            EDEM.from_bytes(rounded.to_bytes()).to_bytes(), rounded.to_bytes())
        independent = FaultGroup(GroupKind.INDEPENDENT, [2], [-0.0])
        exclusive = FaultGroup(GroupKind.EXCLUSIVE, [0, 1], [0.2, 0.3])
        a = EDEM(faults=[FaultDistribution(3, [exclusive, independent])])
        b = EDEM(faults=[FaultDistribution(3, [independent, exclusive])])
        self.assertEqual(a.to_bytes(), b.to_bytes())
        self.assertEqual(a.faults[0].groups[1].probabilities, [0.2, 0.3])
        self.assertEqual(
            struct.pack("d", a.faults[0].groups[0].probabilities[0]),
            struct.pack("d", 0.0))
        for ids, probabilities in (([0, 1], [0.6, 0.5]), ([0, 0], [0.1, 0.2]),
                                   ([0, 1], [float("nan"), 0.2]), ([0,
                                                                    1], [0.1])):
            with self.assertRaises(ValueError):
                EDEM(faults=[
                    FaultDistribution(
                        2,
                        [FaultGroup(GroupKind.EXCLUSIVE, ids, probabilities)])
                ])
        # Separate fault ports preserve independent occurrence-local laws.
        self.assertEqual(
            EDEM(faults=[
                FaultDistribution.independent([0.1]),
                FaultDistribution.independent([0.2])
            ]).vector_sizes, (2, 0))

    def test_closed_and_empty_models(self):
        value = EDEM(faults=[FaultDistribution.independent([0.1])],
                     detectors=[1],
                     observables=[1],
                     successes=[1],
                     blocks=[
                         MatrixBlock(Endpoint(kind), Endpoint(PortKind.FAULT),
                                     SparseMatrix(1, 1, [(0, 0)]))
                         for kind in (PortKind.DETECTOR, PortKind.OBSERVABLE,
                                      PortKind.SUCCESS)
                     ])
        self.assertEqual(value.apply([1]), [1, 1, 1])
        self.assertEqual(EDEM.from_bytes(EDEM().to_bytes()).apply([]), [])
        self.assertEqual(
            EDEM(faults=[FaultDistribution.independent([])]).apply([]), [])

    def test_rejects_untrusted_packed_content(self):
        packed = open_round().to_bytes()
        for size in range(len(packed)):
            with self.assertRaises(ValueError):
                EDEM.from_bytes(packed[:size])
        for field, value in (("magic", bytes(8)), ("version", 0xffffffff),
                             ("directory_count", 0xffffffffffffffff)):
            broken = bytearray(packed)
            HEADER.pack_into(broken, 0, field, value)
            with self.assertRaises(ValueError):
                EDEM.from_bytes(bytes(broken))
        with self.assertRaises(ValueError):
            open_round().apply([0] * 7)


@unittest.skipIf(_adoption_test is None, "native producer is build-tree only")
class AdoptionTest(unittest.TestCase):

    def test_composition_keeps_temporary_sequence_values_alive(self):
        tracker = _adoption_test.Tracker()

        class Occurrences:

            def __len__(self):
                return 2

            def __getitem__(self, index):
                if index >= 2:
                    raise IndexError(index)
                return EDEM.adopt(_adoption_test.make_capsule(tracker))

        result = compose_chain(Occurrences())
        gc.collect()
        self.assertEqual(tracker.result_deallocations, 2)
        self.assertEqual(tracker.live_results, 0)
        self.assertEqual(result.apply([0, 1] + [0] * 8),
                         [0, 0, 1, 0, 1, 0, 1, 0, 0, 0])

    def test_composition_does_not_retain_source_allocation(self):
        tracker = _adoption_test.Tracker()
        capsule = _adoption_test.make_capsule(tracker)
        value = EDEM.adopt(capsule)
        result = compose_chain([value, value])
        del value, capsule
        gc.collect()
        self.assertEqual(tracker.live_results, 0)
        self.assertEqual(tracker.result_deallocations, 1)
        self.assertEqual(tracker.capsule_destructions, 1)
        self.assertEqual(result.apply([0, 1] + [0] * 8),
                         [0, 0, 1, 0, 1, 0, 1, 0, 0, 0])
        self.assertEqual(len(result.faults), 2)

    def test_adopts_native_result_without_copy_and_consumes_once(self):
        tracker = _adoption_test.Tracker()
        capsule = _adoption_test.make_capsule(tracker)
        address = _adoption_test.payload_address(capsule)
        self.assertGreaterEqual(tracker.native_moves, 4)
        self.assertEqual(tracker.result_allocations, 1)
        value = EDEM.adopt(capsule)
        self.assertEqual(value._data_address, address)
        self.assertTrue(_adoption_test.source_empty(capsule))
        self.assertTrue(_adoption_test.consumed(capsule))
        with self.assertRaisesRegex(ValueError, "already been consumed"):
            EDEM.adopt(capsule)
        del capsule
        gc.collect()
        self.assertEqual(tracker.capsule_destructions, 0)
        self.assertEqual(value.to_bytes(), open_round().to_bytes())
        for bits in itertools.product((0, 1), repeat=8):
            f0, f1, v0, v1, s0, s1, l0, l1 = bits
            self.assertEqual(
                value.apply(bits),
                [v0 ^ s0 ^ f0, v1 ^ s1, s0 ^ f0, s1, s0 ^ f1, s1, l0, l1])
        del value
        gc.collect()
        self.assertEqual(tracker.live_results, 0)
        self.assertEqual(tracker.result_deallocations, 1)
        self.assertEqual(tracker.capsule_destructions, 1)
        self.assertEqual(tracker.unconsumed_releases, 0)
        self.assertEqual(tracker.allocations, tracker.deallocations)
        self.assertEqual(tracker.invalid_frees, 0)

    def test_keeps_callback_context_alive_until_payload_release(self):
        tracker = _adoption_test.Tracker()
        tracker_reference = weakref.ref(tracker)
        capsule = _adoption_test.make_capsule(tracker)
        value = EDEM.adopt(capsule)
        del capsule, tracker
        gc.collect()
        self.assertIsNotNone(tracker_reference())
        self.assertEqual(value.apply([0, 1, 0, 0, 0, 0, 0, 0]),
                         [0, 0, 0, 0, 1, 0, 0, 0])
        del value
        gc.collect()
        self.assertIsNone(tracker_reference())

    def test_invalid_content_leaves_source_usable(self):
        tracker = _adoption_test.Tracker()
        capsule = _adoption_test.make_capsule(tracker, corrupt=True)
        address = _adoption_test.payload_address(capsule)
        with self.assertRaises(ValueError):
            EDEM.adopt(capsule)
        self.assertFalse(_adoption_test.consumed(capsule))
        self.assertFalse(_adoption_test.source_empty(capsule))
        self.assertEqual(_adoption_test.payload_address(capsule), address)
        self.assertEqual(tracker.live_results, 1)
        self.assertEqual(tracker.result_deallocations, 0)
        _adoption_test.repair(capsule)
        value = EDEM.adopt(capsule)
        self.assertEqual(value._data_address, address)
        self.assertEqual(value.to_bytes(), open_round().to_bytes())
        del value, capsule
        gc.collect()
        self.assertEqual(tracker.result_deallocations, 1)
        self.assertEqual(tracker.invalid_frees, 0)

    def test_unconsumed_and_unsupported_capsules_release_on_destruction(self):
        for wrong_name in (False, True):
            tracker = _adoption_test.Tracker()
            capsule = _adoption_test.make_capsule(tracker,
                                                  wrong_name=wrong_name)
            if wrong_name:
                with self.assertRaisesRegex(ValueError, "owned-buffer capsule"):
                    EDEM.adopt(capsule)
            self.assertEqual(tracker.live_results, 1)
            del capsule
            gc.collect()
            self.assertEqual(tracker.result_deallocations, 1)
            self.assertEqual(tracker.capsule_destructions, 1)
            self.assertEqual(tracker.unconsumed_releases, 1)
            self.assertEqual(tracker.invalid_frees, 0)

    def test_default_allocator_capsule_and_from_bytes_copy(self):
        tracker = _adoption_test.Tracker()
        capsule = _adoption_test.make_capsule(tracker, use_default=True)
        address = _adoption_test.payload_address(capsule)
        value = EDEM.adopt(capsule)
        copied = EDEM.from_bytes(value.to_bytes())
        self.assertEqual(value._data_address, address)
        self.assertNotEqual(copied._data_address, address)
        self.assertEqual(copied.to_bytes(), value.to_bytes())
        del capsule, value
        gc.collect()
        self.assertEqual(copied.vector_sizes, (8, 8))
        self.assertEqual(tracker.capsule_destructions, 1)

    def test_producer_allocation_failure_releases_temporary_storage(self):
        baseline = _adoption_test.Tracker()
        capsule = _adoption_test.make_capsule(baseline)
        allocation_sites = baseline.attempts
        del capsule
        gc.collect()
        self.assertGreater(allocation_sites, 0)
        self.assertEqual(baseline.allocations, baseline.deallocations)
        for fail_at in range(1, allocation_sites + 1):
            tracker = _adoption_test.Tracker()
            tracker.fail_at = fail_at
            with self.assertRaises(MemoryError):
                _adoption_test.make_capsule(tracker)
            self.assertEqual(tracker.attempts, fail_at)
            self.assertEqual(tracker.allocations, tracker.deallocations)
            self.assertEqual(tracker.live_results, 0)
            self.assertEqual(tracker.invalid_frees, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--require-adoption", action="store_true")
    options, arguments = parser.parse_known_args()
    if options.require_adoption and _adoption_test is None:
        parser.error("native adoption producer is required")
    unittest.main(argv=[sys.argv[0]] + arguments)
