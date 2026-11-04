# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Verify instrument signs, quantum/classical correlations, and ownership."""

from __future__ import annotations

import json

import pytest

import cudaq.logical as ql
from cudaq.logical.errors import GadgetVerificationError

BarePatch = ql.patch[ql.codes.BareQubit]


@ql.codes.code
class TwoLogical:
    block = ql.codes.CSSBlock(data=2)
    hx = ()
    hz = ()
    lx = ((0,), (1,))
    lz = ((0,), (1,))


@ql.codes.code
class BareWithScratch:
    block = ql.codes.CSSBlock(data=1, sx=1)
    hx = ()
    hz = ()
    lx = ((0,),)
    lz = ((0,),)


ScratchPatch = ql.patch[BareWithScratch]


def _verify(gadget):
    receipt = ql.gadgets.verify_gadget(ql.gadgets.GadgetContract(gadget))
    assert receipt.detector_contract.ok
    assert receipt.logical_action.status == "verified"
    assert receipt.verified
    return receipt


def _reject_action(gadget):
    contract = ql.gadgets.GadgetContract(gadget)
    with pytest.raises(GadgetVerificationError) as error:
        ql.gadgets.verify_gadget(contract)
    receipt = error.value.receipt
    # These examples have valid detector contracts. The logical instrument is
    # what distinguishes the good and bad circuits.
    assert receipt.detector_contract.ok
    assert receipt.logical_action.status == "failed"
    assert not receipt.verified
    inspected = ql.gadgets.verify_gadget(contract, raise_on_failure=False)
    assert inspected == receipt
    return receipt


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_builtin_preparation_verifies_the_prepared_eigenstate(basis):
    action = ql.logical.prepare_plus if basis == "X" else ql.logical.prepare_zero

    @ql.gadget(implements=action)
    def prepare(block: BarePatch) -> BarePatch:
        block = ql.reset(block.data)
        return ql.h(block.data) if basis == "X" else block

    _verify(prepare)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_shipped_preparation_factories_verify_their_declared_state(basis):
    factory = ql.gadgets.prepare_plus if basis == "X" else ql.gadgets.prepare_zero
    _verify(factory(ql.codes.BareQubit))


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("wrong", ["sign", "basis"])
def test_preparation_rejects_wrong_sign_and_basis(basis, wrong):
    action = ql.logical.prepare_plus if basis == "X" else ql.logical.prepare_zero

    @ql.gadget(implements=action)
    def prepare(block: BarePatch) -> BarePatch:
        block = ql.reset(block.data)
        if basis == "X":
            block = ql.h(block.data)
        if wrong == "basis":
            return ql.h(block.data)
        return ql.z(block.data) if basis == "X" else ql.x(block.data)

    _reject_action(prepare)


@ql.action
def negative_z_state() -> ql.types.logical_qubit:
    return ql.x(ql.prepare())


@pytest.mark.parametrize("correct_sign", [False, True])
def test_custom_preparation_body_defines_the_sign_and_zero_input_boundary(
        correct_sign):

    @ql.gadget(implements=negative_z_state)
    def prepare(block: BarePatch) -> BarePatch:
        block = ql.reset(block.data)
        return ql.x(block.data) if correct_sign else block

    assert not tuple(prepare.inputs)
    assert len(tuple(prepare.outputs)) == 1
    (_verify if correct_sign else _reject_action)(prepare)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_builtin_and_factory_destructive_readout(basis):
    action = ql.logical.measure_x if basis == "X" else ql.logical.measure_z

    @ql.gadget(implements=action)
    def readout(block: BarePatch) -> bool:
        block, outcome = ql.ops.measure_pauli(block.data,
                                              paulis=basis,
                                              record="readout")
        ql.discard(block)
        return ql.ops.parity(outcome)

    _verify(readout)
    _verify(ql.gadgets.logical_measure(ql.codes.BareQubit, basis=basis))


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_readout_rejects_wrong_basis_despite_identical_result_marginals(basis):
    action = ql.logical.measure_x if basis == "X" else ql.logical.measure_z
    other = "Z" if basis == "X" else "X"

    @ql.gadget(implements=action)
    def readout(block: BarePatch) -> bool:
        block, outcome = ql.ops.measure_pauli(block.data,
                                              paulis=other,
                                              record="readout")
        ql.discard(block)
        return ql.ops.parity(outcome)

    # On the Choi input both measurement results are unbiased. Only their
    # correlation with the inaccessible reference identifies the correct basis.
    _reject_action(readout)


@pytest.mark.parametrize("compensate", [False, True])
def test_affine_result_complement_is_part_of_the_instrument(compensate):

    @ql.gadget(implements=ql.logical.measure_z)
    def readout(block: BarePatch) -> bool:
        if compensate:
            block = ql.x(block.data)
        block, bits = ql.ops.mz(block.data, record="readout")
        outcome = ql.ops.all_zero(bits)
        ql.discard(block)
        return outcome

    # all_zero on one measurement bit returns 1 ^ bit. An X immediately
    # before measurement cancels that affine complement for destructive Z.
    (_verify if compensate else _reject_action)(readout)
    assert readout.outcome_map.constants == (1,)


def test_readout_rejects_a_negated_measurement():

    @ql.gadget(implements=ql.logical.measure_z)
    def readout(block: BarePatch) -> bool:
        block, outcome = ql.ops.measure_pauli(block.data,
                                              paulis="-Z",
                                              record="readout")
        ql.discard(block)
        return ql.ops.parity(outcome)

    _reject_action(readout)


@pytest.mark.parametrize("return_coin", [False, True])
def test_hidden_scratch_randomness_does_not_replace_the_logical_result(
        return_coin):

    @ql.gadget(implements=ql.logical.measure_z)
    def readout(block: ScratchPatch) -> bool:
        block = ql.reset(block.sx)
        block = ql.h(block.sx)
        block, coin = ql.ops.measure_pauli(block.sx, paulis="Z", record="coin")
        block, outcome = ql.ops.measure_pauli(block.data,
                                              paulis="Z",
                                              record="readout")
        ql.discard(block)
        return ql.ops.parity(coin if return_coin else outcome)

    # A discarded independent coin is harmless; substituting it for the
    # returned measurement loses input/result correlation while preserving the
    # uniform result distribution on the Choi input.
    (_reject_action if return_coin else _verify)(readout)


def test_extra_hidden_incompatible_measurement_changes_destructive_readout():

    @ql.gadget(implements=ql.logical.measure_z)
    def readout(block: BarePatch) -> bool:
        block, _ = ql.ops.measure_pauli(block.data,
                                        paulis="X",
                                        record="ignored")
        block, outcome = ql.ops.measure_pauli(block.data,
                                              paulis="Z",
                                              record="readout")
        ql.discard(block)
        return ql.ops.parity(outcome)

    _reject_action(readout)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_preserving_factory_returns_the_measured_logical_and_untouched_spectator(
        basis):
    _verify(
        ql.gadgets.logical_measure(TwoLogical,
                                   basis=basis,
                                   logical=1,
                                   preserve_block=True))


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_destructive_factory_measures_the_selected_logical_and_discards_the_block(
        basis):
    _verify(
        ql.gadgets.logical_measure(TwoLogical,
                                   basis=basis,
                                   logical=1,
                                   preserve_block=False))


@pytest.mark.parametrize("damage", ["reset", "hidden_x_measurement"])
def test_preserving_readout_checks_reference_and_post_measurement_correlations(
        damage):
    intent = ql.gadgets.logical_measure(ql.codes.BareQubit,
                                        basis="z",
                                        preserve_block=True).implements

    @ql.gadget(implements=intent)
    def readout(block: BarePatch) -> tuple[BarePatch, bool]:
        if damage == "hidden_x_measurement":
            block, _ = ql.ops.measure_pauli(block.data,
                                            paulis="X",
                                            record="ignored")
        block, outcome = ql.ops.measure_pauli(block.data,
                                              paulis="Z",
                                              record="readout")
        if damage == "reset":
            block = ql.reset(block.data)
        return block, ql.ops.parity(outcome)

    _reject_action(readout)


def test_preserving_readout_rejects_a_hidden_measurement_on_the_spectator():
    intent = ql.gadgets.logical_measure(TwoLogical,
                                        basis="z",
                                        logical=1,
                                        preserve_block=True).implements

    @ql.gadget(implements=intent,
               logical_ports={
                   "q0": "block.q0",
                   "q1": "block.q1"
               })
    def readout(
            block: ql.patch[TwoLogical]) -> tuple[ql.patch[TwoLogical], bool]:
        block, _ = ql.ops.measure_pauli(block.data[0],
                                        paulis="Z",
                                        record="spectator")
        block, outcome = ql.ops.measure_pauli(block.data[1],
                                              paulis="Z",
                                              record="readout")
        return block, ql.ops.parity(outcome)

    _reject_action(readout)


@ql.action
def keep_first(left: ql.types.logical_qubit,
               right: ql.types.logical_qubit) -> ql.types.logical_qubit:
    ql.discard((right,))
    return left


def test_instrument_outside_the_supported_subset_requires_explicit_inspection():

    @ql.gadget(implements=keep_first)
    def discard_one(left: BarePatch, right: BarePatch) -> BarePatch:
        ql.discard(right)
        return left

    contract = ql.gadgets.GadgetContract(discard_one)
    with pytest.raises(GadgetVerificationError):
        ql.gadgets.verify_gadget(contract)
    receipt = ql.gadgets.verify_gadget(contract, raise_on_failure=False)
    assert receipt.detector_contract.ok
    assert receipt.logical_action.status == "unsupported"
    assert not receipt.verified


def test_entrypoint_without_declared_intent_cannot_receive_verified_status():

    @ql.gadget
    def entrypoint() -> None:
        return None

    contract = ql.gadgets.GadgetContract(entrypoint)
    with pytest.raises(GadgetVerificationError):
        ql.gadgets.verify_gadget(contract)
    receipt = ql.gadgets.verify_gadget(contract, raise_on_failure=False)
    assert receipt.detector_contract.ok
    assert receipt.logical_action.status == "unsupported"
    assert not receipt.verified


@ql.action
def read_second_keep_first(
        left: ql.types.logical_qubit,
        right: ql.types.logical_qubit) -> tuple[ql.types.logical_qubit, bool]:
    outcome = ql.ops.measure_z(right)
    return left, outcome


def test_measurement_with_partial_discard_is_explicitly_unsupported():

    @ql.gadget(implements=read_second_keep_first)
    def readout(left: BarePatch, right: BarePatch) -> tuple[BarePatch, bool]:
        right, outcome = ql.ops.measure_pauli(right.data,
                                              paulis="Z",
                                              record="readout")
        ql.discard(right)
        return left, ql.ops.parity(outcome)

    contract = ql.gadgets.GadgetContract(readout)
    with pytest.raises(GadgetVerificationError):
        ql.gadgets.verify_gadget(contract)
    receipt = ql.gadgets.verify_gadget(contract, raise_on_failure=False)
    assert receipt.detector_contract.ok
    assert receipt.logical_action.status == "unsupported"
    assert not receipt.verified


@pytest.mark.parametrize("reverse_physical", [False, True])
def test_clifford_verification_respects_explicit_reversed_port_binding(
        reverse_physical):

    @ql.gadget(implements=ql.logical.cx,
               logical_ports={
                   "control": "right.q0",
                   "target": "left.q0"
               })
    def cnot(left: BarePatch, right: BarePatch) -> tuple[BarePatch, BarePatch]:
        if reverse_physical:
            right, left = ql.cx(right.data, left.data)
        else:
            left, right = ql.cx(left.data, right.data)
        return left, right

    (_verify if reverse_physical else _reject_action)(cnot)


@pytest.mark.parametrize("damage", [None, "wrong_target", "spectator"])
def test_partial_clifford_binding_preserves_the_other_logical(damage):

    @ql.gadget(implements=ql.logical.h, logical_ports={"q0": "block.q1"})
    def hadamard(block: ql.patch[TwoLogical]) -> ql.patch[TwoLogical]:
        block = ql.h(block.data[0 if damage == "wrong_target" else 1])
        return ql.x(block.data[0]) if damage == "spectator" else block

    (_verify if damage is None else _reject_action)(hadamard)


@pytest.mark.parametrize("basis", ["x", "z"])
def test_logical_pauli_factory_binds_a_selected_port_in_a_multi_logical_block(
        basis):
    _verify(ql.gadgets.logical_pauli(TwoLogical, basis=basis, logical=1))


@pytest.mark.parametrize("action, diagnostic", [
    ({
        "arity": 0,
        "matrix": [],
        "phases": []
    }, "arity"),
    ({
        "arity": 1,
        "matrix": [[1, 0], [1, 0]],
        "phases": [0, 0]
    }, "symplectic"),
])
def test_native_action_payload_rejects_empty_or_nonsymplectic_maps(
        action, diagnostic):
    from cudaq.mlir._mlir_libs import _qlxRuntime

    gadget = ql.gadgets.logical_pauli(ql.codes.BareQubit, basis="x")
    build = ql.compile(ql.gadgets.GadgetContract(gadget))
    with pytest.raises(RuntimeError, match=diagnostic):
        _qlxRuntime.verify_gadget_json(build._fresh_module(), build.root.symbol,
                                       json.dumps(action))


def test_native_objective_resolution_rejects_a_consumed_quantum_return():
    from cudaq.mlir._mlir_libs import _qlxRuntime

    gadget = ql.gadgets.logical_measure(ql.codes.BareQubit,
                                        basis="z",
                                        preserve_block=True)
    build = ql.compile(ql.gadgets.GadgetContract(gadget))
    module = build._fresh_module()
    body = next(op for op in module.body.operations
                if op.operation.name == "qlx.objective_body")
    block = body.regions[0].blocks[0]
    terminator = block.operations[-1].operation
    # Bypass Python's linear-value checks deliberately: the original argument
    # was consumed by MPP, so returning it must not supply trusted intent.
    terminator.operands[0] = block.arguments[0]
    with pytest.raises(RuntimeError, match="consumed|repeated"):
        _qlxRuntime.verify_gadget_json(module, build.root.symbol,
                                       json.dumps({"source": "gadget_spec"}))
