# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Preserve caller ownership when resolving nested compiled ideal bodies."""

from __future__ import annotations

import pytest

import cudaq.logical as ql

BarePatch = ql.patch[ql.codes.BareQubit]


@ql.action
def helper_h(qubit: ql.types.logical_qubit) -> ql.types.logical_qubit:
    return ql.h(qubit)


@pytest.mark.parametrize("calls", [1, 2])
def test_nested_ideal_helper_preserves_a_live_caller_spectator(calls):

    @ql.action
    def intent(
        left: ql.types.logical_qubit, right: ql.types.logical_qubit
    ) -> tuple[ql.types.logical_qubit, ql.types.logical_qubit]:
        for _ in range(calls):
            left = ql.ops.apply(helper_h, left)
        return left, right

    @ql.gadget(implements=intent)
    def realization(left: BarePatch,
                    right: BarePatch) -> tuple[BarePatch, BarePatch]:
        for _ in range(calls):
            left = ql.h(left.data)
        return left, right

    receipt = ql.gadgets.verify_gadget(ql.gadgets.GadgetContract(realization))
    assert receipt.verified
    assert receipt.logical_action.logical_rank == 4
