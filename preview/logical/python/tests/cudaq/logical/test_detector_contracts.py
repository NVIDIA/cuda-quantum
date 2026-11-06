# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import pytest

import cudaq.logical as cql
import cudaq.mlir.ir as ir
from cudaq.mlir._mlir_libs import _qlxRuntime


@cql.gadget(implements=cql.logical.idle)
def memory(block: cql.patch[cql.codes.Steane]) -> cql.patch[cql.codes.Steane]:
    block, _ = cql.extract_syndrome(block, record="round")
    return block


@cql.gadget(implements=cql.logical.measure_z)
def readout(block: cql.patch[cql.codes.Steane]) -> bool:
    block, result = cql.ops.mpp(cql.algebra.Z(block[0]))
    cql.discard(block)
    return result


def _profile_module():
    build = cql.compile(cql.gadgets.GadgetContract(readout))
    module = _qlxRuntime.clone_module(build.module)
    profile = next(op for op in module.body.operations
                   if op.operation.name == "fabric.gadget_contract")
    observable = next(op for op in profile.regions[0].blocks[0].operations
                      if op.operation.name == "fabric.observable")
    return module, observable


def test_detector_map_uses_records_and_virtual_syndromes_in_separate_rows():
    graph = cql.gadgets.GadgetContractBuilder(memory.interface)
    incoming = memory.interface.inputs.only().syndrome
    outgoing = memory.interface.outputs.only().syndrome
    records = tuple(
        memory.record(f"round.s{index}") for index in range(incoming.width))
    for index, record in enumerate(records):
        graph.detector(incoming[index] ^ record)
    graph.bind(outgoing, records)
    profile = graph.freeze()
    assert len(profile.detectors) == incoming.width == 6
    assert profile.detectors[0].parity.input_syndromes == (incoming[0],)
    assert profile.output_syndromes[0].parity.records == (records[0],)
    assert profile.output_syndromes[0].parity.input_syndromes == ()
    build = cql.compile(profile)
    assert build.module.operation.verify()
    assert build.to_mlir().count("fabric.detector") == 6
    assert build.to_mlir().count("fabric.output_syndrome") == 6
    projection = cql.compiler.project_clifford(build, gadget="memory")
    assert all(record.name in projection.record_indices for record in records)
    counts = cql.estimate(build)
    original = cql.estimate(cql.compile(memory))
    assert counts.operation_counts == original.operation_counts
    assert counts.total_operations == original.total_operations


def test_redundant_and_constant_detector_rows_are_declarations_not_proofs():
    detector = cql.gadgets.Detector(memory.record("round.s0"))
    profile = cql.gadgets.GadgetContract(memory,
                                         detectors=(detector, detector,
                                                    cql.gadgets.Detector(False),
                                                    cql.gadgets.Detector(True)))
    # Shape verification deliberately does not certify that any row vanishes
    # ideally, nor that the declared family spans the full detector space.
    assert cql.compile(profile).module.operation.verify()
    assert len(profile.detectors) == 4


def test_observable_rows_are_derived_from_the_gadgets_outcome_map():
    profile = cql.gadgets.GadgetContract(readout)
    assert profile.observables == (cql.gadgets.LogicalObservable(
        readout.record("mpp0.outcome"), 0),)
    build = cql.compile(profile)
    assert "fabric.observable" in build.to_mlir()
    assert build.module.operation.verify()
    assert cql.estimate(build).total_operations == cql.estimate(
        cql.compile(readout)).total_operations
    # Destructive logical Z readout may be random; schema validation must not
    # demand that this outcome be an ideal deterministic detector.
    assert cql.compiler.project_clifford(
        build, gadget="readout").interface.outputs == ()


def test_declared_observables_cannot_override_action_outcomes():
    wrong = cql.gadgets.ProfileParity(records=(readout.record("mpp0.outcome"),),
                                      constant=True)
    with pytest.raises(cql.gadgets.ProfileSemanticError, match="disagrees"):
        cql.gadgets.GadgetContract(readout,
                                   observables=(cql.gadgets.LogicalObservable(
                                       wrong, 0),))


def test_observable_rows_cannot_depend_on_incoming_virtual_syndromes():
    incoming = readout.interface.inputs.only().syndrome[0]
    with pytest.raises(ValueError, match="cannot use input virtual syndromes"):
        cql.gadgets.LogicalObservable(incoming ^ readout.record("mpp0.outcome"),
                                      0)
    module, observable = _profile_module()
    observable.attributes["input_syndromes"] = ir.Attribute.parse(
        "[{port_index = 0 : i64, index = 0 : i64}]", context=module.context)
    with pytest.raises(RuntimeError,
                       match="cannot use input virtual syndromes"):
        _qlxRuntime.project_clifford_json(module, "readout")


@pytest.mark.parametrize("indices", [(1,), (0, 0), (0, 2)])
def test_observable_indices_must_be_unique_and_contiguous(indices):
    with pytest.raises(ValueError, match="unique and contiguous"):
        cql.gadgets.GadgetContract(
            memory,
            observables=tuple(
                cql.gadgets.LogicalObservable(False, index)
                for index in indices))


@pytest.mark.parametrize("index", [-1, True])
def test_observable_indices_are_nonnegative_integers(index):
    with pytest.raises(TypeError, match="index must be nonnegative"):
        cql.gadgets.LogicalObservable(False, index)


def test_profile_references_are_checked_against_the_selected_gadget():
    with pytest.raises(ValueError, match="records from its gadget"):
        cql.gadgets.GadgetContract(memory,
                                   detectors=(cql.gadgets.Detector(
                                       readout.record("mpp0.outcome")),))
    invalid = cql.gadgets.InputSyndromeRef(memory.interface.inputs.only(), 6)
    with pytest.raises(ValueError, match="invalid input syndrome"):
        cql.gadgets.GadgetContract(memory,
                                   detectors=(cql.gadgets.Detector(invalid),))
    profile = cql.gadgets.GadgetContract(memory,
                                         detectors=(cql.gadgets.Detector(
                                             memory.record("missing.s0")),))
    with pytest.raises(ir.MLIRError, match="not an exact output"):
        cql.compile(profile)


@pytest.mark.parametrize("attribute, value, diagnostic", [
    ("idx", "1 : i64", "contiguous in profile order"),
    ("constant", "true", "disagrees with linked GadgetSpec outcome_map"),
    ("records", '["readout.missing.outcome"]', "not an exact output"),
])
def test_native_verifier_rechecks_observable_structure(attribute, value,
                                                       diagnostic):
    module, observable = _profile_module()
    observable.attributes[attribute] = ir.Attribute.parse(
        value, context=module.context)
    with pytest.raises(RuntimeError, match=diagnostic):
        _qlxRuntime.project_clifford_json(module, "readout")


def test_native_verifier_handles_a_missing_observable_index():
    module, observable = _profile_module()
    del observable.attributes["idx"]
    with pytest.raises(RuntimeError,
                       match="observable index|requires attribute 'idx'"):
        _qlxRuntime.project_clifford_json(module, "readout")


def test_contract_compilation_and_replay_preserve_ir_and_stage():
    contract = cql.gadgets.GadgetContract(readout)
    build = cql.compile(contract)
    assert build.root.kind == "gadget_contract"
    assert build.profile == "p2a"
    assert "fabric.gadget_contract" in build.to_mlir()
    contract_op = next(op for op in build.module.body.operations
                       if op.operation.name == "fabric.gadget_contract")
    assert contract_op.regions[0].blocks[0].operations[
        -1].operation.name == "fabric.contract_end"
    replayed = cql.compiler.Build.replay(build.serialize())
    assert replayed.root == build.root
    assert replayed.profile == build.profile
    assert replayed.to_mlir() == build.to_mlir()


def test_contract_api_has_one_authoring_builder_and_no_legacy_aliases():
    from cudaq.logical.gadgets.contracts import GadgetContract, GadgetContractBuilder
    from cudaq.logical.gadgets.contract_lowerer import GadgetContractLowerer

    assert cql.gadgets.GadgetContract is GadgetContract
    assert cql.gadgets.GadgetContractBuilder is GadgetContractBuilder
    assert cql.gadgets.GadgetContractLowerer is GadgetContractLowerer
    for old_name in ("GadgetProfile", "ProfileGraph", "GadgetProfileBuilder"):
        assert not hasattr(cql.gadgets, old_name)
    assert not hasattr(cql.compiler, "GadgetProfileBuilder")
    assert not hasattr(cql.compiler, "GadgetContractBuilder")
