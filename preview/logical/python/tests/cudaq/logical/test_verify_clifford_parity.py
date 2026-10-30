# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Check signed Clifford phases and quantum-result order independently."""

from __future__ import annotations

import json

import pytest

import cudaq.logical as ql

BarePatch = ql.patch[ql.codes.BareQubit]


@ql.action
def signed_clifford(
    left: ql.types.logical_qubit, right: ql.types.logical_qubit
) -> tuple[ql.types.logical_qubit, ql.types.logical_qubit]:
    left = ql.h(left)
    left = ql.s(left)
    left, right = ql.cx(left, right)
    right = ql.sdg(right)
    left, right = ql.cz(left, right)
    right = ql.h(right)
    return left, right


@ql.gadget(implements=signed_clifford)
def signed_clifford_realization(
        left: BarePatch, right: BarePatch) -> tuple[BarePatch, BarePatch]:
    left = ql.h(left.data)
    left = ql.s(left.data)
    left, right = ql.cx(left.data, right.data)
    right = ql.sdg(right.data)
    left, right = ql.cz(left.data, right.data)
    right = ql.h(right.data)
    return left, right


# Independently conjugating X0, X1, Z0, Z1 through H0 S0 CX01 SDag1 CZ01 H1
# gives +Z0, +Z0Y1, -X0Z1, +Z0X1. These fixed images were also checked with
# Stim, which is not needed to run this test. Columns are X0, X1, Z0, Z1.
_SIGNED_IMAGES = ((0, 0, 1, 0), (0, 1, 1, 1), (1, 0, 0, 1), (0, 1, 1, 0))
_SIGNED_PHASES = (0, 0, 1, 0)


@pytest.mark.parametrize("wrong_sign", [False, True])
def test_native_signed_clifford_matches_independent_pauli_images(wrong_sign):
    from cudaq.mlir._mlir_libs import _qlxRuntime

    build = ql.compile(ql.gadgets.GadgetContract(signed_clifford_realization))
    phases = list(_SIGNED_PHASES)
    if wrong_sign:
        phases[2] ^= 1
    expected = {"arity": 2, "matrix": _SIGNED_IMAGES, "phases": phases}
    receipt = json.loads(
        _qlxRuntime.verify_gadget_json(build._fresh_module(), build.root.symbol,
                                       json.dumps(expected)))
    assert receipt["detector_contract"]["ok"]
    logical = receipt["logical_action"]
    assert logical["status"] == ("failed" if wrong_sign else "verified")
    assert logical["rows"] == [
        "verified", "verified", "sign_mismatch" if wrong_sign else "verified",
        "verified"
    ]


def test_compiled_objective_resolves_the_independent_signed_clifford():
    receipt = ql.gadgets.verify_gadget(
        ql.gadgets.GadgetContract(signed_clifford_realization))
    assert receipt.verified
    assert receipt.logical_action.verified_checks == len(_SIGNED_IMAGES)


@ql.action
def reverse_quantum_results(
    left: ql.types.logical_qubit, right: ql.types.logical_qubit
) -> tuple[ql.types.logical_qubit, ql.types.logical_qubit]:
    return right, left


@pytest.mark.parametrize("reverse_results", [False, True])
def test_clifford_quantum_results_follow_the_objective_order(reverse_results):

    @ql.gadget(implements=reverse_quantum_results)
    def realization(left: BarePatch,
                    right: BarePatch) -> tuple[BarePatch, BarePatch]:
        return (right, left) if reverse_results else (left, right)

    receipt = ql.gadgets.verify_gadget(ql.gadgets.GadgetContract(realization),
                                       raise_on_failure=False)
    assert receipt.detector_contract.ok
    assert receipt.verified == reverse_results
    assert receipt.logical_action.status == ("verified"
                                             if reverse_results else "failed")


@ql.action
def measure_left_reverse_quantum_results(
    left: ql.types.logical_qubit, right: ql.types.logical_qubit
) -> tuple[ql.types.logical_qubit, ql.types.logical_qubit, bool]:
    left, outcome = ql.ops.mpp(ql.algebra.Z(left))
    return right, left, outcome


@pytest.mark.parametrize("reverse_results", [False, True])
def test_preserving_readout_quantum_results_follow_the_objective_order(
        reverse_results):

    @ql.gadget(implements=measure_left_reverse_quantum_results)
    def realization(left: BarePatch,
                    right: BarePatch) -> tuple[BarePatch, BarePatch, bool]:
        left, outcome = ql.ops.mpp(ql.algebra.Z(left[0]))
        return (right, left, outcome) if reverse_results else (left, right,
                                                               outcome)

    receipt = ql.gadgets.verify_gadget(ql.gadgets.GadgetContract(realization),
                                       raise_on_failure=False)
    assert receipt.detector_contract.ok
    assert receipt.verified == reverse_results
    assert receipt.logical_action.status == ("verified"
                                             if reverse_results else "failed")
