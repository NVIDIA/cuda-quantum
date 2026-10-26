# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import pytest

import cudaq.logical as ql
from cudaq.logical.errors import GadgetVerificationError

RepetitionPatch = ql.patch[ql.codes.Repetition]
SteanePatch = ql.patch[ql.codes.Steane]


def _repetition_round(action):

    @ql.gadget(implements=action)
    def syndrome_round(block: RepetitionPatch) -> RepetitionPatch:
        block, _ = ql.extract_syndrome(block)
        return block

    return syndrome_round


syndrome_round = _repetition_round(ql.logical.idle)
block_in = syndrome_round.inputs.only()
block_out = syndrome_round.outputs.only()
checks = syndrome_round.records.syndrome(0).checks


def _rows(incoming, new_checks, order=None):
    """One detector per syndrome component: input syndrome ^ new check."""

    order = range(incoming.syndrome.width) if order is None else order
    return tuple(
        ql.gadgets.Detector(incoming.syndrome[i] ^ new_checks[j])
        for i, j in enumerate(order))


def _contract(gadget=syndrome_round, *, detectors=None):
    incoming = gadget.inputs.only()
    outgoing = gadget.outputs.only()
    new_checks = gadget.records.syndrome(0).checks
    if detectors is None:
        detectors = _rows(incoming, new_checks)
    return ql.gadgets.GadgetProfile(gadget,
                                    detectors=detectors,
                                    boundary={outgoing.syndrome: new_checks})


def _kinds(error):
    receipt = error.value.receipt
    return ([f.kind for f in receipt.detector_contract.failures],
            [f.kind for f in receipt.logical_action.failures])


def test_repetition_round_receipt_certifies_contract_and_action():
    receipt = ql.gadgets.verify_gadget(_contract())
    contract = receipt.detector_contract
    assert contract.ok
    assert (contract.outcome_code_rank, contract.detector_rank,
            contract.deterministic_logical_outcome_rank) == (4, 4, 0)
    assert contract.outcome_count == 6
    assert (contract.detector_count, contract.output_relation_count) == (2, 2)
    assert receipt.logical_action.status == "verified"
    assert receipt.logical_action.action == "ql.logical.idle"
    assert receipt.logical_action.logical_rank == 2
    assert receipt.verified


def test_contract_is_separate_from_the_gadget_body():
    # The same gadget verifies under a redundant contract: rows may repeat.
    redundant = (*_rows(block_in, checks),
                 ql.gadgets.Detector(block_in.syndrome[0] ^ checks[0]))
    receipt = ql.gadgets.verify_gadget(_contract(detectors=redundant))
    assert receipt.detector_contract.detector_count == 3
    assert receipt.detector_contract.detector_rank == 4


def test_omitting_a_detector_fails_the_rank_condition():
    partial = (ql.gadgets.Detector(block_in.syndrome[0] ^ checks[0]),)
    with pytest.raises(GadgetVerificationError, match="rank condition") as exc:
        ql.gadgets.verify_gadget(_contract(detectors=partial))
    contract = exc.value.receipt.detector_contract
    assert not contract.ok
    assert (contract.outcome_code_rank, contract.detector_rank,
            contract.deterministic_logical_outcome_rank) == (4, 3, 0)
    # The action is independent of the contract and still verifies.
    assert exc.value.receipt.logical_action.status == "verified"
    assert _kinds(exc) == (["rank_incomplete"], [])


def test_swapped_records_make_both_detectors_nondeterministic():
    swapped = _rows(block_in, checks, order=(1, 0))
    with pytest.raises(GadgetVerificationError) as exc:
        ql.gadgets.verify_gadget(_contract(detectors=swapped))
    assert _kinds(exc)[0] == ["detector_not_deterministic"] * 2


def test_wrong_pauli_action_passes_contract_and_fails_with_sign_mismatch():
    wrong = _repetition_round(ql.logical.x)
    with pytest.raises(GadgetVerificationError,
                       match=r"sign mismatch on Z_ref -Z̄") as exc:
        ql.gadgets.verify_gadget(_contract(wrong))
    receipt = exc.value.receipt
    # The contract is blind to the logical action.
    assert receipt.detector_contract.ok
    assert receipt.logical_action.status == "failed"
    assert receipt.logical_action.logical_rank == 1
    assert _kinds(exc) == ([], ["sign_mismatch"])


def test_wrong_basis_action_is_not_stabilized():
    wrong = _repetition_round(ql.logical.h)
    with pytest.raises(GadgetVerificationError, match="action mismatch") as exc:
        ql.gadgets.verify_gadget(_contract(wrong))
    assert exc.value.receipt.detector_contract.ok
    assert exc.value.receipt.logical_action.logical_rank == 0
    assert _kinds(exc) == ([], ["action_mismatch"] * 2)


def test_failure_can_be_returned_instead_of_raised():
    wrong = _repetition_round(ql.logical.x)
    receipt = ql.gadgets.verify_gadget(_contract(wrong), raise_on_failure=False)
    assert not receipt.verified
    assert receipt.logical_action.failures[0].kind == "sign_mismatch"


def test_steane_transversal_h_swaps_the_check_families():

    @ql.gadget(implements=ql.logical.h)
    def steane_h_round(block: SteanePatch) -> SteanePatch:
        block = ql.h(block.data)
        block, _ = ql.extract_syndrome(block)
        return block

    h_in = steane_h_round.inputs.only()
    h_out = steane_h_round.outputs.only()
    h_checks = steane_h_round.records.syndrome(0).checks
    swapped = tuple(h_in.syndrome[(i + 3) % 6] for i in range(6))
    contract = ql.gadgets.GadgetProfile(
        steane_h_round,
        detectors=tuple(
            ql.gadgets.Detector(swapped[i] ^ h_checks[i]) for i in range(6)),
        boundary={h_out.syndrome: swapped})
    receipt = ql.gadgets.verify_gadget(contract)
    assert (receipt.detector_contract.outcome_code_rank,
            receipt.detector_contract.detector_rank) == (12, 12)
    assert receipt.logical_action.action == "ql.logical.h"

    # The adjacent-round contract does not hold across a Hadamard.
    unswapped = ql.gadgets.GadgetProfile(steane_h_round,
                                         detectors=_rows(h_in, h_checks),
                                         boundary={h_out.syndrome: h_checks})
    with pytest.raises(GadgetVerificationError) as exc:
        ql.gadgets.verify_gadget(unswapped)
    assert _kinds(exc)[0] == ["detector_not_deterministic"] * 6


def test_two_block_gadget_certifies_logical_cx():

    @ql.gadget(implements=ql.logical.cx)
    def cx_round(
            control: RepetitionPatch,
            target: RepetitionPatch) -> tuple[RepetitionPatch, RepetitionPatch]:
        control, target = ql.cx(control.data, target.data)
        control, _ = ql.extract_syndrome(control)
        target, _ = ql.extract_syndrome(target)
        return control, target

    control_in, target_in = cx_round.inputs[0], cx_round.inputs[1]
    control_out, target_out = cx_round.outputs[0], cx_round.outputs[1]
    control_checks = cx_round.records.syndrome(0).checks
    target_checks = cx_round.records.syndrome(1).checks
    contract = ql.gadgets.GadgetProfile(
        cx_round,
        detectors=(
            *_rows(control_in, control_checks),
            *(ql.gadgets.Detector(control_in.syndrome[i] ^
                                  target_in.syndrome[i] ^ target_checks[i])
              for i in range(2))),
        boundary={
            control_out.syndrome: control_checks,
            target_out.syndrome: target_checks
        })
    receipt = ql.gadgets.verify_gadget(contract)
    assert receipt.detector_contract.outcome_code_rank == 8
    assert receipt.logical_action.action == "ql.logical.cx"
    assert receipt.logical_action.logical_rank == 4


def test_non_clifford_action_is_reported_as_unsupported():

    @ql.gadget(implements=ql.logical.t)
    def t_round(block: RepetitionPatch) -> RepetitionPatch:
        block, _ = ql.extract_syndrome(block)
        return block

    receipt = ql.gadgets.verify_gadget(_contract(t_round))
    assert receipt.detector_contract.ok
    assert receipt.logical_action.status == "unsupported"
    assert "non-Clifford" in receipt.logical_action.reason
    assert not receipt.verified


def test_verify_gadget_rejects_a_bare_gadget():
    with pytest.raises(TypeError, match="GadgetProfile"):
        ql.gadgets.verify_gadget(syndrome_round)


def test_verified_contract_derives_an_edem():
    pytest.importorskip("cudaq_edem")
    ql.gadgets.verify_gadget(_contract())
    ql.analysis.derive_edem(ql.compile(_contract()))
