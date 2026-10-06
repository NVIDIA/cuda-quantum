# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Typed P2 non-Clifford magic-state cultivation authoring.

This module traces the pinned Gidney--Shutty d=3 unitary-injection,
cultivation, and escape circuit into ordinary Fabric operations. The authors'
reference is a Cliffordized fault-analysis envelope. Their vector sampler
recovers the physical protocol by interpreting *every* ``S``/``S_DAG`` in the
injection and double-cat check as ``T``/``T_DAG``. We retain that semantics
exactly: one injected quarter turn and fourteen quarter turns implementing the
two logical-H checks.

The preparation gadget stops before the reference circuit's terminal MPP
boundary, leaving a live encoded magic state. TSim execution requires a
separate verified, evidence-bearing P3 device projection; this P2 module does
not claim one.
"""

from __future__ import annotations

from cudaq.logical.ops._impl import (
    all_false,
    allocate_patch,
    cx,
    cz,
    h,
    measure_pauli,
    mz,
    pack_resource,
    parity,
    postselect,
    reset,
    s,
    sdg,
    t,
    tdg,
    tick,
    x,
    z,
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
    ProfileParity,
    SuccessPredicate,
    gadget,
    patch,
)
from cudaq.logical.protocols.definition import protocol
from cudaq.logical.types.semantic import resource
from ..std import LogicalInstrumentRef, T_STATE, produce
from ._cultivation_fixture import (
    FRAME_SIZE,
    OUTPUT_CARRIERS,
    OUTPUT_HX,
    OUTPUT_HZ,
    OUTPUT_LX,
    OUTPUT_LZ,
    REFERENCE_SHA256,
    UPSTREAM_ARCHIVE_MD5,
    UPSTREAM_REVISION,
    reference_stim,
)

CULTIVATED_MATCHABLE_D6 = CSSCode(
    name="cultivated_matchable_d6",
    n=37,
    k=1,
    d=6,
    block=CSSBlock(data=37),
    hx=OUTPUT_HX,
    hz=OUTPUT_HZ,
    lx=OUTPUT_LX,
    lz=OUTPUT_LZ,
)
cultivated_matchable_d6 = CULTIVATED_MATCHABLE_D6.encoding(
    name="cultivated_matchable_d6_encoding",
    layout={"carrier_labels": OUTPUT_CARRIERS},
)

_scratch = tuple(
    carrier for carrier in range(FRAME_SIZE) if carrier not in OUTPUT_CARRIERS)
_boundary_roles = CarrierRoleMap(active=OUTPUT_CARRIERS, scratch=_scratch)
_cultivation_frame = PatchTransform(
    name="cultivation_d3_to_matchable_d6_frame",
    source=cultivated_matchable_d6,
    destination=cultivated_matchable_d6,
    frame=Block(data=FRAME_SIZE),
    source_support=OUTPUT_CARRIERS,
    destination_support=OUTPUT_CARRIERS,
    source_roles=_boundary_roles,
    destination_roles=_boundary_roles,
    logical_map=(0,),
    evidence=f"gidney_shutty_reference_sha256:{REFERENCE_SHA256}",
)

_cultivation_checks_objective = LogicalInstrumentRef(
    "cultivated_t_preparation_checks",
    arity=1,
    result_arity=7,
)
_cultivation_analysis_cache = {}
_CULTIVATION_SUCCESS_RECORDS = (
    (5, 6, 7),
    (7, 15),
    (7, 16),
    (34,),
    (35,),
    (7, 19),
    (37,),
)


def _reference_circuit():
    try:
        import stim
    except ImportError as exc:  # pragma: no cover - dependency diagnostic
        raise ImportError("actual cultivation authoring requires stim") from exc
    return stim.Circuit(reference_stim()).flattened()


def _measure_pauli_group(state, group, *, record):
    """Retain one native physical MPP group in the Fabric circuit."""

    carriers = []
    paulis = []
    invert = False
    for target in group:
        carriers.append(target.qubit_value)
        invert ^= bool(target.is_inverted_result_target)
        if target.is_x_target:
            paulis.append("X")
        elif target.is_y_target:
            paulis.append("Y")
        elif target.is_z_target:
            paulis.append("Z")
        else:  # pragma: no cover - fixture audit
            raise ValueError(f"unsupported MPP target {target!r}")
    if invert:
        raise ValueError(
            "inverted physical MPP is not present in the pinned fixture")
    return measure_pauli(state.frame[tuple(carriers)],
                         paulis="".join(paulis),
                         record=record)


def _trace_actual_cultivation(state):
    records = []
    t_targets = 0
    tdg_targets = 0
    output_boundary_seen = False

    for instruction in _reference_circuit():
        name = instruction.name
        targets = instruction.targets_copy()
        qubits = tuple(
            target.qubit_value for target in targets if target.is_qubit_target)

        if name == "TICK":
            tick()
            continue
        if name in {"QUBIT_COORDS", "SHIFT_COORDS"}:
            continue
        if name == "MPP":
            output_boundary_seen = True
            break

        if name == "R":
            state = reset(state.frame[qubits])
        elif name == "RX":
            state = reset(state.frame[qubits])
            state = h(state.frame[qubits])
        elif name in {"H", "S", "S_DAG", "X", "Z"}:
            # This is the defining non-Clifford interpretation used by the
            # authors' VecInterceptSampler. The seven rotations before and
            # after the cat check are part of the check, not ordinary S gates.
            operation = {
                "H": h,
                "S": t,
                "S_DAG": tdg,
                "X": x,
                "Z": z,
            }[name]
            state = operation(state.frame[qubits])
            if name == "S":
                t_targets += len(qubits)
            elif name == "S_DAG":
                tdg_targets += len(qubits)
        elif name in {"CX", "CZ"}:
            pairs = tuple(
                (targets[index].qubit_value, targets[index + 1].qubit_value)
                for index in range(0, len(targets), 2))
            controls, destinations = zip(*pairs)
            operation = cx if name == "CX" else cz
            state = operation(state.frame[controls], state.frame[destinations])
        elif name in {"M", "MZ", "MX", "MY", "MPP"}:
            for group in instruction.target_groups():
                record = f"cultivation_m{len(records)}"
                if name == "MPP":
                    all_y = all(target.is_y_target for target in group)
                    carriers = tuple(target.qubit_value for target in group)
                    if all_y:
                        # VecInterceptSampler's magic-basis MPP conjugation.
                        state = tdg(state.frame[carriers])
                        state = s(state.frame[carriers])
                    state, bits = _measure_pauli_group(state,
                                                       group,
                                                       record=record)
                    if all_y:
                        state = sdg(state.frame[carriers])
                        state = t(state.frame[carriers])
                else:
                    (target,) = group
                    carrier = (target.qubit_value,)
                    if name == "MX":
                        state = h(state.frame[carrier])
                    elif name == "MY":
                        state = sdg(state.frame[carrier])
                        state = h(state.frame[carrier])
                    state, bits = mz(state.frame[carrier], record=record)
                records.append(bits)
        elif targets and all(
                target.is_measurement_record_target for target in targets):
            # Reference-only annotations are not part of the product circuit.
            continue
        else:  # pragma: no cover - fixture audit
            raise ValueError(f"unsupported cultivation instruction {name!r}")

    if not output_boundary_seen:
        raise ValueError(
            "pinned cultivation fixture has no terminal output boundary")
    if (t_targets, tdg_targets) != (7, 8):
        raise ValueError(
            "expected the audited 15 quarter-turn cultivation construction "
            f"(7 T, 8 T-dagger), found {(t_targets, tdg_targets)}")
    selected_events = tuple(
        parity(*(records[index]
                 for index in support))
        for support in _CULTIVATION_SUCCESS_RECORDS)
    return state, tuple(selected_events)


def _cultivation_analysis(gadget):
    """Build the detached success table for the pinned fixture."""

    cached = _cultivation_analysis_cache.get(gadget)
    if cached is not None:
        return cached

    success = tuple(
        SuccessPredicate(
            ProfileParity(records=tuple(
                gadget.record(f"cultivation_m{index}.data0")
                for index in support)))
        for support in _CULTIVATION_SUCCESS_RECORDS)
    analysis = GadgetProfile(
        gadget,
        success=success,
        name="cultivated_t_production_analysis",
    )
    _cultivation_analysis_cache[gadget] = analysis
    return analysis


@gadget(
    implements=_cultivation_checks_objective,
    transform=_cultivation_frame,
    metadata={
        "protocol": "Gidney-Shutty magic-state cultivation",
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_archive_md5": UPSTREAM_ARCHIVE_MD5,
        "reference_sha256": REFERENCE_SHA256,
        "injection": "unitary T-dagger",
        "double_cat_check": "14 quarter-turn controlled-H realization",
        "non_clifford_rotation_count": 15,
        "cultivation_distance": 3,
        "output_distance": 6,
    },
)
def prepare_cultivated_t(
    candidate: patch[cultivated_matchable_d6],
) -> tuple[patch[cultivated_matchable_d6], bool, bool, bool, bool, bool, bool,
           bool]:
    """Prepare, cultivate, and escape one physical T state."""

    candidate, failures = _trace_actual_cultivation(candidate)
    return (candidate, *failures)


@protocol(
    implements=produce(T_STATE, code=cultivated_matchable_d6),
    metadata={
        "protocol": "actual_cultivation_d3_to_matchable_d6",
        "selection": "seven detached success parities",
    },
)
def cultivate_t_d3_to_matchable_d6() -> resource[T_STATE]:
    candidate = allocate_patch(cultivated_matchable_d6,
                               region="cultivation_factory")
    candidate, *failures = prepare_cultivated_t(
        candidate,
        analysis=_cultivation_analysis(prepare_cultivated_t),
    )
    accepted = all_false(*failures)
    postselect(
        accepted,
        expected=True,
    )
    return pack_resource(candidate, kind=T_STATE)


__all__ = [
    "CULTIVATED_MATCHABLE_D6",
    "cultivated_matchable_d6",
    "cultivate_t_d3_to_matchable_d6",
    "prepare_cultivated_t",
]
