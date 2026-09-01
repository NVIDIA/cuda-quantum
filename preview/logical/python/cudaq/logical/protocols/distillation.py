# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Concrete five-qubit 15-to-1 T-state distillation.

The implementation is the compressed triorthogonal circuit described by
Litinski: four single-qubit rotations are absorbed into four raw magic-state
inputs and the remaining eleven columns become Z-product pi/4 rotations on
five qubits.  The four even rows are measured in X and postselected; the odd
row is the output.
"""

from __future__ import annotations

import math

from .. import codes, std
from ..programs.decorators import objective
from ..ops._impl import (
    allocate_patch,
    discard,
    h,
    measure_x,
    measure_z,
    mz,
    pack_resource,
    parity,
    postselect,
    prepare_plus,
    request_many,
    resource_rotate,
    rotate,
    s,
    unpack_resource,
)
from ..types.values import logical_qubit
from ..algebra.pauli import Z
from ..gadgets import (
    gadget,
    patch,
)
from ..protocols.definition import protocol
from ..types.semantic import resource

# Columns 4..14 of the standard 15-to-1 triorthogonal matrix.  Columns
# 0..3 are single-qubit rotations on the four check rows and are represented by
# the first four raw input states.
FIFTEEN_TO_ONE_ROTATION_SUPPORTS = (
    (2, 3, 4),
    (1, 3, 4),
    (1, 2, 4),
    (1, 2, 3),
    (0, 3, 4),
    (0, 2, 4),
    (0, 2, 3),
    (0, 1, 4),
    (0, 1, 3),
    (0, 1, 2),
    (0, 1, 2, 3, 4),
)


def _make_rotation_step(index: int, support: tuple[int, ...]):
    label = "".join(map(str, support))

    @objective(name=f"z_product_pi_over_4_{index}_{label}")
    def intent(
        q0: logical_qubit,
        q1: logical_qubit,
        q2: logical_qubit,
        q3: logical_qubit,
        q4: logical_qubit,
    ) -> tuple[
            logical_qubit,
            logical_qubit,
            logical_qubit,
            logical_qubit,
            logical_qubit,
    ]:
        values = [q0, q1, q2, q3, q4]
        product = Z(values[support[0]])
        for port in support[1:]:
            product = product @ Z(values[port])
        updated = rotate(product, angle=math.pi / 4.0)
        for port, value in zip(support, updated):
            values[port] = value
        return tuple(values)

    @gadget(implements=intent, name=f"bare_raw_t_z_product_{index}_{label}")
    def realization(
        q0: patch[codes.BareQubit],
        q1: patch[codes.BareQubit],
        q2: patch[codes.BareQubit],
        q3: patch[codes.BareQubit],
        q4: patch[codes.BareQubit],
        raw: resource[std.RAW_T_STATE],
    ) -> tuple[
            patch[codes.BareQubit],
            patch[codes.BareQubit],
            patch[codes.BareQubit],
            patch[codes.BareQubit],
            patch[codes.BareQubit],
    ]:
        values = [q0, q1, q2, q3, q4]
        product = Z(values[support[0]][0])
        for port in support[1:]:
            product = product @ Z(values[port][0])
        updated = resource_rotate(raw, product, angle=math.pi / 4.0)
        for port, value in zip(support, updated):
            values[port] = value
        return tuple(values)

    return realization


FIFTEEN_TO_ONE_ROTATION_STEPS = tuple(
    _make_rotation_step(index, support)
    for index, support in enumerate(FIFTEEN_TO_ONE_ROTATION_SUPPORTS))


@objective
def bare_s_action(q: logical_qubit) -> logical_qubit:
    return s(q)


@gadget(implements=bare_s_action)
def bare_s(block: patch[codes.BareQubit]) -> patch[codes.BareQubit]:
    return s(block.data)


@objective
def bare_measure_x_intent(q: logical_qubit) -> bool:
    return measure_x(q)


@gadget(implements=bare_measure_x_intent)
def bare_measure_x(block: patch[codes.BareQubit]) -> bool:
    block = h(block.data)
    block, bits = mz(block.data, record="check_x")
    outcome = parity(bits)
    discard(block)
    return outcome


@protocol(implements=std.produce(std.T_STATE))
def distill_15to1() -> resource[std.T_STATE]:
    raw = request_many(std.RAW_T_STATE, count=15)
    output = prepare_plus(
        allocate_patch(codes.BareQubit, region="t_state_factory"))

    checks = []
    for state in raw[:4]:
        output, check = unpack_resource(state,
                                        like=output,
                                        encoding=codes.BareQubit)
        checks.append(check)

    values = [*checks, output]
    for state, rotation in zip(raw[4:], FIFTEEN_TO_ONE_ROTATION_STEPS):
        values = list(rotation(*values, state))

    # The standard positive-angle triorthogonal identity produces T-dagger on
    # the odd row. Logical S converts it to the canonical T|+> resource.
    values[4] = bare_s(values[4])

    for check in values[:4]:
        postselect(bare_measure_x(check), expected=False)

    return pack_resource(values[4], kind=std.T_STATE)


__all__ = [
    "FIFTEEN_TO_ONE_ROTATION_SUPPORTS",
    "FIFTEEN_TO_ONE_ROTATION_STEPS",
    "bare_measure_x",
    "bare_s",
    "distill_15to1",
]
