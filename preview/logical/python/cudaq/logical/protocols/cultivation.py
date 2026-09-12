# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Typed color-code growth used by magic-state cultivation.

The reusable gadget contains the physical d=3 to d=5 growth circuit from the
Gidney--Shutty cultivation construction.  A separate validation gadget wraps
the same circuit body with boundary Pauli measurements.  The wrapper is a
test/evidence artifact; those measurements are not part of production growth.
"""

from __future__ import annotations

from cudaq.logical.ops._impl import (
    all_false,
    cx,
    h,
    mz,
    pack_resource,
    parity,
    reset,
    retry,
)
from cudaq.logical.codes import (
    Block,
    CSSBlock,
    CSSCode,
    CarrierRoleMap,
    PatchTransform,
)
from cudaq.logical.gadgets import (
    GadgetProfile,
    GadgetSpec,
    OutcomeMap,
    Port,
    OutcomeRole,
    RetryExhaustion,
    before_resource_output,
    gadget,
    patch,
)
from cudaq.logical.algebra.gf2 import GF2Matrix
from cudaq.logical.protocols.definition import protocol
from cudaq.logical.types.semantic import resource
from ..std import LogicalInstrumentRef, T_STATE, produce
from ..gadgets.success import all_zero as success_all_zero

COLOR_3_CARRIERS = (0, 1, 2, 3, 4, 5, 8)
COLOR_5_CARRIERS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    9,
    10,
    12,
    13,
    15,
    17,
    18,
    20,
    21,
    22,
    23,
)

_color_3_checks = (
    (0, 4, 2, 1),
    (1, 2, 5, 3),
    (4, 8, 5, 2),
)
_color_5_checks = (
    (0, 4, 2, 1),
    (8, 21, 18, 13),
    (5, 9, 15, 10, 6, 3),
    (13, 18, 22, 20, 15, 9),
    (1, 2, 5, 3),
    (6, 10, 17, 12),
    (4, 8, 13, 9, 5, 2),
    (15, 20, 17, 10),
    (21, 23, 22, 18),
)


def _local_rows(carriers, rows):
    positions = {carrier: index for index, carrier in enumerate(carriers)}
    return tuple(tuple(positions[carrier] for carrier in row) for row in rows)


COLOR_3 = CSSCode(
    name="cultivation_color_3",
    n=7,
    k=1,
    d=3,
    block=CSSBlock(data=7),
    hx=_local_rows(COLOR_3_CARRIERS, _color_3_checks),
    hz=_local_rows(COLOR_3_CARRIERS, _color_3_checks),
    lx=(tuple(range(7)),),
    lz=(tuple(range(7)),),
)
COLOR_5 = CSSCode(
    name="cultivation_color_5",
    n=19,
    k=1,
    d=5,
    block=CSSBlock(data=19),
    hx=_local_rows(COLOR_5_CARRIERS, _color_5_checks),
    hz=_local_rows(COLOR_5_CARRIERS, _color_5_checks),
    lx=(tuple(range(19)),),
    lz=(tuple(range(19)),),
)

color_3 = COLOR_3.encoding(
    name="cultivation_color_3_encoding",
    layout={"carrier_labels": COLOR_3_CARRIERS},
)
color_5 = COLOR_5.encoding(
    name="cultivation_color_5_encoding",
    layout={"carrier_labels": COLOR_5_CARRIERS},
)


def _boundary_roles(frame_size, support):
    support = tuple(support)
    return CarrierRoleMap(
        active=support,
        dormant=tuple(
            index for index in range(frame_size) if index not in support),
    )


color_3_to_5 = PatchTransform(
    name="cultivation_color_3_to_5",
    source=color_3,
    destination=color_5,
    frame=Block(data=24),
    source_support=COLOR_3_CARRIERS,
    destination_support=COLOR_5_CARRIERS,
    source_roles=_boundary_roles(24, COLOR_3_CARRIERS),
    destination_roles=_boundary_roles(24, COLOR_5_CARRIERS),
    logical_map=(0,),
    evidence="cultivation_growth_boundary_flows_v1",
)

_growth_zero = (9, 13, 18, 22, 23, 12, 6, 11, 10, 17, 20)
_growth_plus = (14, 19, 21, 7, 16, 15)
_growth_layers = (
    ((14, 9), (19, 22), (21, 23), (15, 20), (16, 10), (7, 11)),
    ((14, 13), (19, 18), (16, 17), (11, 12), (7, 6)),
    ((13, 14), (18, 19), (17, 16), (12, 11), (6, 7)),
)
_growth_measured = (14, 19, 16, 11, 7)
_growth_objective = LogicalInstrumentRef(
    "cultivation_growth_projection",
    arity=1,
    result_arity=len(_growth_measured),
)
_growth_records = tuple(
    f"growth{index}.data0" for index in range(len(_growth_measured)))
_growth_spec = GadgetSpec(
    implements=_growth_objective,
    ports=(Port(
        "state",
        direction="inout",
        encoding=color_5,
        logical_ports={"q0": "q0"},
    ),),
    record_schema=_growth_records,
    outcome_map=OutcomeMap(
        records=_growth_records,
        matrix=GF2Matrix.from_rows(
            tuple(
                tuple(
                    int(row == column)
                    for column in range(len(_growth_records)))
                for row in range(len(_growth_records)))),
        roles=tuple((OutcomeRole.SUCCESS,) for _ in _growth_records),
    ),
)


def _apply_color_3_to_5_growth(state):
    """Apply the physical circuit without changing the Python owner."""

    state = reset(state.frame[_growth_zero])
    state = reset(state.frame[_growth_plus])
    state = h(state.frame[_growth_plus])
    for pairs in _growth_layers:
        controls, targets = zip(*pairs)
        # These are two disjoint views of the same linear owner. Pairwise view
        # order carries the directed interaction relation; this is not a CX
        # from the frame onto itself.
        state = cx(state.frame[controls], state.frame[targets])
    outcomes = []
    for index, carrier in enumerate(_growth_measured):
        state, bits = mz(state.frame[(carrier,)], record=f"growth{index}")
        outcomes.append(parity(bits))
    return state, tuple(outcomes)


@gadget(
    spec=_growth_spec,
    transform=color_3_to_5,
)
def grow_color_3_to_5(
    state: patch[color_3],
) -> tuple[patch[color_5], bool, bool, bool, bool, bool]:
    """Grow one encoded color-code patch while preserving its logical state."""

    state, outcomes = _apply_color_3_to_5_growth(state)
    return (state, *outcomes)


grow_color_3_to_5_profile = GadgetProfile(
    grow_color_3_to_5,
    success=success_all_zero(
        tuple(grow_color_3_to_5.record(name) for name in _growth_records),),
    name="grow_color_3_to_5_analysis",
)


@protocol(
    implements=produce(T_STATE, code=color_5),
    metadata={
        "stage": "cultivation_growth",
        "input": "encoded_color_3_T_candidate",
        "output_encoding": "cultivation_color_5_encoding",
        "selection": "five_growth_projection_checks",
    },
)
def cultivate_color_3_to_5(candidate: patch[color_3],) -> resource[T_STATE]:
    """Grow and commit an accepted encoded T-state candidate.

    The caller supplies the injected d=3 candidate.  Failed projection checks
    replay the selection-bearing growth attempt.  Only the accepted d=5 patch
    crosses the commit boundary into the resource flow.
    """

    candidate, *projection_failures = grow_color_3_to_5(
        candidate,
        analysis=grow_color_3_to_5_profile,
    )
    accepted = all_false(*projection_failures)
    candidate = retry(
        candidate,
        until=accepted,
        max_attempts=8,
        exhaustion=RetryExhaustion.ABORT,
        commit_point=before_resource_output(),
    )
    return pack_resource(candidate, kind=T_STATE)


__all__ = [
    "COLOR_3",
    "COLOR_5",
    "COLOR_3_CARRIERS",
    "COLOR_5_CARRIERS",
    "color_3",
    "color_5",
    "color_3_to_5",
    "grow_color_3_to_5",
    "grow_color_3_to_5_profile",
    "cultivate_color_3_to_5",
]
