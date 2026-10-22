# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Circuit-derived EDEM responses and ideal detector-contract evidence."""

from itertools import product
import ctypes
import gc
import json

import pytest

import cudaq.logical as cql
import cudaq.mlir.ir as ir
from cudaq.mlir._mlir_libs import _qlxRuntime
from cudaq.logical.analysis import ElementaryFault, FaultSet, PauliInsertion, derive_edem

import cudaq_edem as edem


def _row(paulis):
    return tuple(int(p in "XY") for p in paulis) + tuple(
        int(p in "ZY") for p in paulis)


# This presentation makes X on carrier 0 the first pure syndrome error:
# S=(1,0), L=(0,0). It matches the open-round equation in the EDEM fixture.
repetition = cql.codes.Code(name="edem_repetition",
                            n=3,
                            k=1,
                            stabilizers=tuple(map(_row, ("ZZI", "IZZ"))),
                            logicals=((_row("XXX"), _row("IZI")),),
                            anti_stabilizers=tuple(map(_row, ("XII", "IIX"))))


@cql.gadget(implements=cql.logical.h)
def hadamard(
        block: cql.patch[cql.codes.BareQubit]
) -> cql.patch[cql.codes.BareQubit]:
    return cql.h(block.data)


@cql.gadget(implements=cql.logical.cx)
def cnot(
    left: cql.patch[cql.codes.BareQubit], right: cql.patch[cql.codes.BareQubit]
) -> tuple[cql.patch[cql.codes.BareQubit], cql.patch[cql.codes.BareQubit]]:
    return cql.cx(left.data, right.data)


@cql.gadget(implements=cql.logical.idle)
def round_checks(block: cql.patch[repetition]) -> cql.patch[repetition]:
    block, _ = cql.ops.measure_pauli(block.data[(0, 1)],
                                     paulis="ZZ",
                                     record="first")
    block, _ = cql.ops.measure_pauli(block.data[(1, 2)],
                                     paulis="ZZ",
                                     record="second")
    return block


@cql.gadget(implements=cql.logical.measure_z)
def readout(block: cql.patch[cql.codes.BareQubit]) -> bool:
    block, result = cql.ops.mpp(cql.algebra.Z(block[0]))
    cql.discard(block)
    return result


def _round_profile(*, detectors=True, wrong_output=False):
    graph = cql.gadgets.ProfileGraph(round_checks.interface)
    incoming = round_checks.interface.inputs.only().syndrome
    outgoing = round_checks.interface.outputs.only().syndrome
    records = (round_checks.record("first.outcome"),
               round_checks.record("second.outcome"))
    if detectors:
        for index, record in enumerate(records):
            graph.detector(incoming[index] ^ record)
    graph.bind(outgoing, (records[1], records[0]) if wrong_output else records)
    return graph.freeze()


def _native_owned_record(capsule):
    """Read an actual producer's live C record; never fabricate native data."""

    class OwnedBuffer(ctypes.Structure):
        _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_uint64),
                    ("deallocate", ctypes.c_void_p),
                    ("context", ctypes.c_void_p)]

    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
    get_pointer.restype = ctypes.c_void_p
    pointer = get_pointer(capsule, b"cudaq_edem.owned_buffer.v0")
    return OwnedBuffer.from_address(pointer)


def test_native_owned_result_adopts_once_without_payload_copy():
    build = cql.compile(cql.gadgets.GadgetProfile(hadamard))
    module = build._fresh_module()
    capsule = _qlxRuntime.derive_edem_owned(module, "", FaultSet().to_json())
    record = _native_owned_record(capsule)
    address, size = record.data, record.size
    assert address and size
    model = edem.EDEM.adopt(capsule)
    assert model._data_address == address
    assert (record.data, record.size, record.deallocate,
            record.context) == (None, 0, None, None)
    with pytest.raises(ValueError, match="already been consumed"):
        edem.EDEM.adopt(capsule)
    # The checked value owns bytes independently of its source module/Build.
    del record, capsule, module, build
    gc.collect()
    for x, z in product((0, 1), repeat=2):
        assert model.apply([x, z]) == [z, x]


def test_derive_edem_adopts_the_native_allocation(monkeypatch):
    native = _qlxRuntime.derive_edem_owned
    addresses = []

    def capture(*args):
        capsule = native(*args)
        addresses.append(_native_owned_record(capsule).data)
        return capsule

    monkeypatch.setattr(_qlxRuntime, "derive_edem_owned", capture)
    model = derive_edem(cql.compile(cql.gadgets.GadgetProfile(hadamard)))
    assert addresses == [model._data_address]
    assert model.apply([1, 0]) == [0, 1]


def test_hadamard_transfer_and_fault_sites_all_inputs():
    build = cql.compile(cql.gadgets.GadgetProfile(hadamard))
    before = build.to_mlir(), build.content_sha256
    projection = cql.compiler.project_clifford(build, gadget="hadamard")
    faults = FaultSet().pauli_error("", 0, "x", .01).pauli_error(
        projection.instructions[0].site_id, 0, "x", .02)
    model = derive_edem(build, faults=faults)
    assert isinstance(model, edem.EDEM)
    assert model.vector_sizes == (4, 2)
    for f0, f1, x, z in product((0, 1), repeat=4):
        assert model.apply([f0, f1, x, z]) == [z ^ f1, x ^ f0]
    assert (build.to_mlir(), build.content_sha256) == before
    assert derive_edem(build, faults=faults).to_bytes() == model.to_bytes()
    assert edem.EDEM.from_bytes(model.to_bytes()).to_bytes() == model.to_bytes()
    # The producer uses the private authenticated snapshot.
    inspection = build.module
    gadget = next(op for op in inspection.body.operations
                  if op.operation.name == "fabric.gadget")
    gate = next(op for op in gadget.regions[0].blocks[0].operations
                if op.operation.name == "fabric.h")
    gate.attributes["indices"] = ir.DenseI64ArrayAttr.get(
        [99], context=inspection.context)
    assert derive_edem(build, faults=faults).to_bytes() == model.to_bytes()


def test_cnot_preserves_separate_code_block_order():
    model = derive_edem(cql.compile(cql.gadgets.GadgetProfile(cnot)))
    assert len(model.inputs) == len(model.outputs) == 2
    assert len(model.codes) == 1
    for x0, z0, x1, z1 in product((0, 1), repeat=4):
        assert model.apply([x0, z0, x1, z1]) == [x0, z0 ^ z1, x1 ^ x0, z1]


def test_open_round_exhaustive_separates_virtual_syndrome_and_boundary_error():
    build = cql.compile(_round_profile())
    projection = cql.compiler.project_clifford(build, gadget="round_checks")
    faults = FaultSet().record_flip(round_checks.record("first.outcome"),
                                    .01).pauli_error(
                                        projection.instructions[-1].site_id, 0,
                                        "x", .02)
    model = derive_edem(build, faults=faults)
    assert model.vector_sizes == (8, 8)
    assert model.detectors == [2]
    for f0, f1, v0, v1, s0, s1, lx, lz in product((0, 1), repeat=8):
        assert model.apply(
            [f0, f1, v0, v1, s0, s1, lx,
             lz]) == [v0 ^ s0 ^ f0, v1 ^ s1, s0 ^ f0, s1, s0 ^ f1, s1, lx, lz]
    for block in model.blocks:
        if block.input.kind == edem.PortKind.VIRTUAL:
            assert block.output.kind in (edem.PortKind.DETECTOR,
                                         edem.PortKind.VIRTUAL)


def test_preparation_has_only_output_and_readout_allows_random_observables():
    prepare = cql.gadgets.prepare_zero(cql.codes.BareQubit)
    build = cql.compile(cql.gadgets.GadgetProfile(prepare))
    projection = cql.compiler.project_clifford(build, gadget=prepare.name)
    faults = FaultSet().pauli_error(projection.instructions[-1].site_id, 0, "x",
                                    .1)
    model = derive_edem(build, faults=faults)
    assert model.inputs == []
    assert model.apply([1]) == [1, 0]
    build = cql.compile(cql.gadgets.GadgetProfile(readout))
    model = derive_edem(build,
                        faults=FaultSet().record_flip("mpp0.outcome", .1))
    assert model.outputs == []
    assert model.observables == [1]
    for fault, x, z in product((0, 1), repeat=3):
        assert model.apply([fault, x, z]) == [fault ^ x]


def test_rank_incomplete_and_incorrect_output_contracts_are_rejected():
    # All output assignments are present, yet the omitted input/record
    # relations leave the contract semantically incomplete.
    profile = _round_profile(detectors=False)
    assert profile.boundary_complete
    with pytest.raises(RuntimeError, match="rank-incomplete"):
        derive_edem(cql.compile(profile))
    with pytest.raises(RuntimeError, match="ideal output stabilizer"):
        derive_edem(cql.compile(_round_profile(wrong_output=True)))


def test_nondeterministic_detectors_and_wrong_affine_constants_are_rejected():
    profile = cql.gadgets.GadgetProfile(readout,
                                        detectors=(cql.gadgets.Detector(
                                            readout.record("mpp0.outcome")),))
    with pytest.raises(RuntimeError, match="not deterministically zero"):
        derive_edem(cql.compile(profile))
    build = cql.compile(_round_profile())
    module = _qlxRuntime.clone_module(build.module)
    profile = next(op for op in module.body.operations
                   if op.operation.name == "fabric.gadget_profile")
    assignment = next(op for op in profile.regions[0].blocks[0].operations
                      if op.operation.name == "fabric.output_syndrome")
    assignment.attributes["constant"] = ir.BoolAttr.get(True,
                                                        context=module.context)
    with pytest.raises(RuntimeError, match="ideal output stabilizer"):
        _qlxRuntime.derive_edem_owned(module, "", FaultSet().to_json())


def test_redundant_and_zero_detectors_remain_valid():
    profile = _round_profile()
    from dataclasses import replace
    redundant = replace(profile,
                        boundary=None,
                        detectors=(*profile.detectors, profile.detectors[0],
                                   cql.gadgets.Detector(False)))
    model = derive_edem(cql.compile(redundant))
    assert model.detectors == [4]
    assert model.apply([0, 0, 1, 0, 0, 0])[:4] == [1, 0, 1, 0]


def test_categorical_laws_and_correlated_effects_are_preserved():
    build = cql.compile(cql.gadgets.GadgetProfile(hadamard))
    faults = FaultSet().depolarize1("", 0, .3)
    model = derive_edem(build, faults=faults)
    group, = model.faults[0].groups
    assert group.kind == edem.GroupKind.EXCLUSIVE
    assert group.fault_ids == [0, 1, 2]
    assert group.probabilities == pytest.approx([.1] * 3)
    # Three categorical alternatives remain three indicators even when their
    # linear combination has zero effect.
    assert model.apply([1, 1, 1, 0, 0]) == [0, 0]
    correlated = FaultSet((ElementaryFault(
        "pair", .05,
        (PauliInsertion("", ((0, "x"),)), PauliInsertion("op0.0",
                                                         ((0, "z"),)))),))
    assert derive_edem(build, faults=correlated).apply([1, 0, 0]) == [0, 0]
    zero_alternative = FaultSet().pauli_channel("", {
        "x": 0.0,
        "y": .1,
        "z": .2
    }, (0,))
    zero_model = derive_edem(build, faults=zero_alternative)
    assert zero_model.faults[0].groups[0].fault_ids == [0, 1, 2]
    assert zero_model.faults[0].groups[0].probabilities == [0.0, .1, .2]


@pytest.mark.parametrize("faults, diagnostic", [
    (FaultSet().pauli_error("missing", 0, "x", .1), "unknown insertion site"),
    (FaultSet().pauli_error("", 7, "x", .1), "non-live carrier"),
    (FaultSet().record_flip("missing", .1), "unknown stable record"),
])
def test_faults_resolve_against_exact_projection(faults, diagnostic):
    with pytest.raises(RuntimeError, match=diagnostic):
        derive_edem(cql.compile(cql.gadgets.GadgetProfile(hadamard)),
                    faults=faults)


def test_native_fault_parser_rejects_malformed_and_unsupported_inputs():
    module = cql.compile(cql.gadgets.GadgetProfile(hadamard))._fresh_module()
    for change in ({
            "exclusive_groups": [[0, 0]]
    }, {
            "probabilities": []
    }, {
            "extra": True
    }, {
            "faults": [{
                "id": "e",
                "erasure_count": 1
            }]
    }):
        payload = json.loads(FaultSet().pauli_error("", 0, "x", .1).to_json())
        payload.update(change)
        with pytest.raises(RuntimeError, match="invalid EDEM fault input"):
            _qlxRuntime.derive_edem_owned(module, "", json.dumps(payload))


def test_probability_totals_use_the_same_validator_as_edem_construction():
    build = cql.compile(cql.gadgets.GadgetProfile(hadamard))
    faults = FaultSet().pauli_channel("", {"x": .6, "z": .5}, (0,))
    with pytest.raises(RuntimeError, match="EDEM assembly failed"):
        derive_edem(build, faults=faults)
    with pytest.raises(ValueError):
        edem.EDEM(faults=[
            edem.FaultDistribution(
                2,
                [edem.FaultGroup(edem.GroupKind.EXCLUSIVE, [0, 1], [.6, .5])])
        ])


def test_overcomplete_presentation_section_on_attainable_syndromes():
    profile = cql.codes.CodeProfile(repetition,
                                    effective_stabilizers=tuple(
                                        map(_row, ("ZZI", "IZZ", "ZIZ"))))
    encoding = cql.codes.Encoding(repetition, profile=profile)

    @cql.gadget(implements=cql.logical.idle)
    def identity(block: cql.patch[encoding]) -> cql.patch[encoding]:
        return block

    graph = cql.gadgets.ProfileGraph(identity.interface)
    graph.bind(identity.interface.outputs.only().syndrome,
               identity.interface.inputs.only().syndrome)
    model = derive_edem(cql.compile(graph.freeze()))
    assert model.codes[0].generators == 3
    # The third coordinate is redundant. A section is inverse to the CPM on
    # its image; it need not preserve arbitrary inconsistent three-bit inputs.
    for v0, v1, s0, s1, x, z in product((0, 1), repeat=6):
        inputs = [v0, v1, v0 ^ v1, s0, s1, s0 ^ s1, x, z]
        assert model.apply(inputs) == inputs


def test_effective_stabilizer_products_keep_their_affine_sign():
    bell = cql.codes.Code(name="edem_bell",
                          n=2,
                          k=0,
                          stabilizers=tuple(map(_row, ("XX", "ZZ"))),
                          anti_stabilizers=tuple(map(_row, ("ZI", "IX"))))
    profile = cql.codes.CodeProfile(bell,
                                    effective_stabilizers=tuple(
                                        map(_row, ("XX", "ZZ", "YY"))))
    encoding = cql.codes.Encoding(bell, profile=profile)

    @cql.gadget(implements=cql.logical.idle)
    def check_bell(block: cql.patch[encoding]) -> cql.patch[encoding]:
        block, _ = cql.ops.measure_pauli(block.data, paulis="YY", record="yy")
        return block

    incoming = check_bell.interface.inputs.only().syndrome
    graph = cql.gadgets.ProfileGraph(check_bell.interface)
    graph.bind(check_bell.interface.outputs.only().syndrome, incoming)
    graph.detector(incoming[2] ^ check_bell.record("yy.outcome"))
    model = derive_edem(cql.compile(graph.freeze()))
    assert model.detectors == [1]
    # EDEM propagates flips. The - sign belongs to the ideal affine contract,
    # and disappears from the linear response map after validation.
    assert model.apply([0] * 6) == [0] * 7


def test_canonical_and_assembled_syndrome_aliases_are_one_fault_effect():

    @cql.gadget(implements=cql.logical.idle)
    def steane(
            block: cql.patch[cql.codes.Steane]) -> cql.patch[cql.codes.Steane]:
        block, _ = cql.extract_syndrome(block, record="round")
        return block

    incoming = steane.interface.inputs.only().syndrome
    records = tuple(steane.record(f"round.s{i}") for i in range(incoming.width))
    graph = cql.gadgets.ProfileGraph(steane.interface)
    graph.bind(steane.interface.outputs.only().syndrome, records)
    for i, record in enumerate(records):
        graph.detector(incoming[i] ^ record)
    build = cql.compile(graph.freeze())
    projection = cql.compiler.project_clifford(build, gadget="steane")
    canonical = projection.interface.measurements[0].record
    first = derive_edem(build, faults=FaultSet().record_flip(canonical, .1))
    alias = derive_edem(build,
                        faults=FaultSet().record_flip("steane.round.s0", .1))
    assert first.to_bytes() == alias.to_bytes()
    assert first.apply([1] + [0] * 14) == [1] + [0] * 5 + [1] + [0] * 13


def test_same_dimensions_do_not_alias_different_logical_presentations():
    swapped = cql.codes.Code(name="swapped_bare",
                             n=1,
                             k=1,
                             logicals=((_row("Z"), _row("X")),))

    @cql.gadget(implements=cql.logical.idle)
    def idle(block: cql.patch[swapped]) -> cql.patch[swapped]:
        return block

    ordinary = derive_edem(cql.compile(cql.gadgets.GadgetProfile(hadamard)))
    changed = derive_edem(cql.compile(cql.gadgets.GadgetProfile(idle)))
    assert ordinary.codes[0].name != changed.codes[0].name
    assert changed.apply([1, 0]) == [1, 0]


@pytest.mark.parametrize(
    "values", [[.25, .25], [.5, .5000005], [.5, .500002], [1 / 9] * 9])
def test_probability_acceptance_tracks_native_edem_policy(values):
    faults = FaultSet(
        tuple(
            ElementaryFault(str(i), p, (PauliInsertion("", ((0, "x"),)),))
            for i, p in enumerate(values)), (tuple(range(len(values))),))
    build = cql.compile(cql.gadgets.GadgetProfile(hadamard))
    distribution = edem.FaultDistribution(len(values), [
        edem.FaultGroup(edem.GroupKind.EXCLUSIVE, list(range(len(values))),
                        values)
    ])
    try:
        direct = edem.EDEM(faults=[distribution])
    except ValueError:
        with pytest.raises(RuntimeError, match="EDEM assembly failed"):
            derive_edem(build, faults=faults)
    else:
        produced = derive_edem(build, faults=faults)
        assert produced.faults[0].groups[0].probabilities == direct.faults[
            0].groups[0].probabilities


@pytest.mark.parametrize(
    "value", [-1e-15, 1 + 1e-15,
              float("nan"), float("inf"), True])
def test_individual_probability_bounds_are_strict(value):
    with pytest.raises(ValueError, match="finite in"):
        FaultSet().pauli_error("", 0, "x", value)


def test_measured_carriers_remain_available_and_late_faults_affect_later_reads(
):

    @cql.gadget(implements=cql.logical.measure_z)
    def repeated(block: cql.patch[cql.codes.BareQubit]) -> bool:
        block, _ = cql.mz(block.data[0], record="first")
        block, bits = cql.mz(block.data[0], record="second")
        cql.discard(block)
        return cql.ops.parity(bits)

    profile = cql.gadgets.GadgetProfile(repeated,
                                        detectors=(cql.gadgets.Detector(
                                            repeated.record("first.data0") ^
                                            repeated.record("second.data0")),))
    build = cql.compile(profile)
    projection = cql.compiler.project_clifford(build, gadget="repeated")
    model = derive_edem(build,
                        faults=FaultSet().pauli_error(
                            projection.instructions[0].site_id, 0, "x", .1))
    for fault, x, z in product((0, 1), repeat=3):
        assert model.apply([fault, x, z]) == [fault, fault ^ x]


def test_output_virtual_syndrome_can_follow_realization_randomness():
    fixed = cql.codes.Code(name="edem_fixed",
                           n=1,
                           k=0,
                           stabilizers=(_row("Z"),),
                           anti_stabilizers=(_row("X"),))

    @cql.gadget(implements=cql.logical.idle)
    def random_sign(block: cql.patch[fixed]) -> cql.patch[fixed]:
        block = cql.h(block.data)
        block, _ = cql.ops.measure_pauli(block.data, paulis="Z", record="sign")
        return block

    graph = cql.gadgets.ProfileGraph(random_sign.interface)
    graph.bind(random_sign.interface.outputs.only().syndrome,
               (random_sign.record("sign.outcome"),))
    model = derive_edem(cql.compile(graph.freeze()),
                        faults=FaultSet().record_flip("sign.outcome", .1))
    # The ideal sign is random, but its relation to the declared V_out is
    # deterministic. A record flip changes V_out without a residual Pauli.
    assert model.apply([1, 0, 0]) == [1, 0]


def test_standard_measurement_noise_preserves_later_carrier_effects():
    from cudaq.logical.analysis import standard_depolarizing_faults

    @cql.gadget(implements=cql.logical.measure_z)
    def repeated_noise(block: cql.patch[cql.codes.BareQubit]) -> bool:
        block, _ = cql.mz(block.data[0], record="first")
        block, bits = cql.mz(block.data[0], record="second")
        cql.discard(block)
        return cql.ops.parity(bits)

    profile = cql.gadgets.GadgetProfile(
        repeated_noise,
        detectors=(cql.gadgets.Detector(
            repeated_noise.record("first.data0") ^
            repeated_noise.record("second.data0")),))
    build = cql.compile(profile)
    projection = cql.compiler.project_clifford(build, gadget="repeated_noise")
    faults = standard_depolarizing_faults(
        projection,
        cql.compiler.StandardDepolarizingNoise(
            before_measure_flip_probability=.1))
    assert len(faults.faults) == 2
    assert all(
        fault.insertions and not fault.record_flips for fault in faults.faults)
    model = derive_edem(build, faults=faults)
    for first, second, x, z in product((0, 1), repeat=4):
        assert model.apply([first, second, x,
                            z]) == [second, first ^ second ^ x]
    with pytest.raises(ValueError, match="projected instruction"):
        standard_depolarizing_faults(
            projection,
            cql.compiler.StandardDepolarizingNoise(),
            depolarizations=(cql.compiler.Depolarize1Placement(
                "missing", (0,), .1),))

    # Explicit layers also work at entry and between measurements, without
    # assuming any extraction structure. X/Y before the first measurement flip
    # both records; X/Y before the second flip only the second record.
    first, second = projection.instructions
    placed = standard_depolarizing_faults(
        projection,
        cql.compiler.StandardDepolarizingNoise(),
        depolarizations=(cql.compiler.Depolarize1Placement(
            first.site_id, (0,),
            .3), cql.compiler.Depolarize1Placement(second.site_id, (0,), .6)))
    assert len(placed.faults) == 6
    assert placed.exclusive_groups == ((0, 1, 2), (3, 4, 5))
    model = derive_edem(build, faults=placed)
    for column, expected in enumerate(
        ([0, 1], [0, 1], [0, 0], [1, 1], [1, 1], [0, 0])):
        vector = [0] * model.vector_sizes[0]
        vector[column] = 1
        assert model.apply(vector) == expected


def test_overlapping_detector_and_observable_rows_do_not_complete_missing_rank(
):

    @cql.gadget(implements=cql.logical.measure_z)
    def duplicate_check(block: cql.patch[cql.codes.BareQubit]) -> bool:
        block = cql.ops.prepare_zero(block)
        block, first = cql.mz(block.data[0], record="first")
        block, _ = cql.mz(block.data[0], record="omitted")
        cql.discard(block)
        return cql.ops.parity(first)

    profile = cql.gadgets.GadgetProfile(
        duplicate_check,
        detectors=(cql.gadgets.Detector(
            duplicate_check.record("first.data0")),))
    with pytest.raises(RuntimeError, match="rank-incomplete"):
        derive_edem(cql.compile(profile))


@pytest.mark.parametrize("gate", ["s", "sdg", "x", "z"])
def test_remaining_single_carrier_cliffords_and_pauli_phases(gate):

    @cql.gadget(implements=cql.logical.idle)
    def phase(
        block: cql.patch[cql.codes.BareQubit]
    ) -> cql.patch[cql.codes.BareQubit]:
        return getattr(cql.ops, gate)(block.data)

    model = derive_edem(cql.compile(cql.gadgets.GadgetProfile(phase)))
    for x, z in product((0, 1), repeat=2):
        assert model.apply([x,
                            z]) == [x, z ^ (x if gate in ("s", "sdg") else 0)]


def test_success_bits_have_separate_outputs_and_must_agree_ideally():
    from dataclasses import replace
    profile = _round_profile()
    profile = replace(profile,
                      boundary=None,
                      detectors=(),
                      success=tuple(
                          cql.gadgets.SuccessPredicate(row.parity)
                          for row in profile.detectors))
    model = derive_edem(cql.compile(profile),
                        faults=FaultSet().record_flip("first.outcome", .1))
    assert model.detectors == []
    assert model.successes == [2]
    assert model.apply([1] + [0] * 6) == [1, 0, 1, 0, 0, 0, 0, 0]
    bad = cql.gadgets.GadgetProfile(readout,
                                    success=(cql.gadgets.SuccessPredicate(
                                        readout.record("mpp0.outcome")),))
    with pytest.raises(RuntimeError, match="ideal accepted value"):
        derive_edem(cql.compile(bad))


def test_relative_record_can_start_with_the_gadget_name():

    @cql.gadget(implements=cql.logical.idle)
    def prefixed(
        block: cql.patch[cql.codes.BareQubit]
    ) -> cql.patch[cql.codes.BareQubit]:
        block, _ = cql.ops.measure_pauli(block.data,
                                         paulis="Z",
                                         record="prefixed.read")
        block, _ = cql.ops.measure_pauli(block.data, paulis="Z", record="again")
        return block

    profile = cql.gadgets.GadgetProfile(
        prefixed,
        detectors=(cql.gadgets.Detector(
            prefixed.record("prefixed.read.outcome") ^
            prefixed.record("again.outcome")),))
    build = cql.compile(profile)
    relative = derive_edem(build,
                           faults=FaultSet().record_flip(
                               "prefixed.read.outcome", .1))
    qualified = derive_edem(build,
                            faults=FaultSet().record_flip(
                                "prefixed.prefixed.read.outcome", .1))
    assert relative.to_bytes() == qualified.to_bytes()
    assert relative.apply([1, 0, 0]) == [1, 0, 0]
