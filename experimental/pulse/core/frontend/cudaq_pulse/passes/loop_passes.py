# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loop optimization passes for the pulse IR.

LICM (loop-invariant code motion) and loop strength reduction.
"""

from __future__ import annotations

from .ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    clone_program,
)

# ---------------------------------------------------------------------------
# Loop structure analysis
# ---------------------------------------------------------------------------


def _find_loops(ops: list[Op]) -> list[tuple[int, int]]:
    """Find (start_idx, end_idx) pairs for FOR_LOOP..END_FOR regions."""
    loops: list[tuple[int, int]] = []
    stack: list[int] = []

    for idx, op in enumerate(ops):
        if op.kind in (OpKind.FOR_LOOP, "for_begin"):
            stack.append(idx)
        elif op.kind in (OpKind.END_FOR, "for_end"):
            if not stack:
                raise ValueError(
                    f"END_FOR at op[{idx}] without matching FOR_LOOP")
            start = stack.pop()
            loops.append((start, idx))

    if stack:
        raise ValueError(f"FOR_LOOP at op[{stack}] without matching END_FOR")

    return loops


def _inner_loop_ranges(
        loops: list[tuple[int, int]]) -> dict[tuple[int, int], set[int]]:
    """For each loop, collect indices that belong to strictly inner loops."""
    result: dict[tuple[int, int], set[int]] = {}
    for outer_s, outer_e in loops:
        inner_idx: set[int] = set()
        for inner_s, inner_e in loops:
            if inner_s > outer_s and inner_e < outer_e:
                for i in range(inner_s, inner_e + 1):
                    inner_idx.add(i)
        result[(outer_s, outer_e)] = inner_idx
    return result


def _values_defined_in_range(ops: list[Op], start: int, end: int) -> set[int]:
    """Collect vids of all values defined (produced) within [start, end]."""
    defined: set[int] = set()
    for idx in range(start, end + 1):
        for v in ops[idx].results:
            defined.add(v.vid)
    return defined


_WAVEFORM_OPS = frozenset({
    OpKind.MAKE_WAVEFORM,
    "square",
    "gaussian",
    "drag",
    "cosine",
    "tanh_ramp",
    "gaussian_square",
    "custom",
    "custom_samples",
    "wf_add",
    "wf_sub",
    "wf_mul",
    "wf_scale",
    "wf_neg",
})


def _op_is_loop_invariant(op: Op, loop_defined: set[int]) -> bool:
    """An op is loop-invariant if none of its operands are defined inside the loop."""
    for v in op.operands:
        if v.vid in loop_defined:
            return False
    return True


# ---------------------------------------------------------------------------
# LICM: Loop-Invariant Code Motion
# ---------------------------------------------------------------------------


def run_licm(program: Program) -> Program:
    """Hoist loop-invariant waveform construction out of for loops.

    A waveform op is loop-invariant if all its operands are defined outside
    the loop body.
    """
    result = clone_program(program)
    ops = result.ops

    changed = True
    while changed:
        changed = False
        loops = _find_loops(ops)
        inner_ranges = _inner_loop_ranges(loops)

        for loop_start, loop_end in reversed(loops):
            loop_defined = _values_defined_in_range(ops, loop_start + 1,
                                                    loop_end - 1)
            hoisted: list[Op] = []
            body_ops: list[Op] = []
            skip = inner_ranges.get((loop_start, loop_end), set())

            for idx in range(loop_start + 1, loop_end):
                op = ops[idx]
                if idx in skip:
                    body_ops.append(op)
                elif (op.kind in _WAVEFORM_OPS and
                      _op_is_loop_invariant(op, loop_defined)):
                    hoisted.append(op)
                    for v in op.results:
                        loop_defined.discard(v.vid)
                    changed = True
                else:
                    body_ops.append(op)

            if hoisted:
                new_ops = (ops[:loop_start] + hoisted + [ops[loop_start]] +
                           body_ops + [ops[loop_end]] + ops[loop_end + 1:])
                ops = new_ops
                break

    result.ops = ops
    return result


# ---------------------------------------------------------------------------
# Loop Strength Reduction
# ---------------------------------------------------------------------------


def _detect_linear_phase_progression(
        ops: list[Op],
        loop_start: int,
        loop_end: int,
        skip: set[int] | None = None) -> list[tuple[int, int, float]]:
    """Detect shift_phase ops with constant delta inside the immediate loop body.

    Returns list of (op_index, tone_vid, delta).
    """
    candidates: list[tuple[int, int, float]] = []
    skip = skip or set()

    for idx in range(loop_start + 1, loop_end):
        if idx in skip:
            continue
        op = ops[idx]
        if op.kind not in (OpKind.SHIFT_PHASE, "shift_phase"):
            continue

        delta = op.attrs.get(
            "delta_rad",
            op.attrs.get("phase", op.attrs.get("phase_rad",
                                               op.attrs.get("arg1"))))
        if delta is None:
            continue

        try:
            delta_f = float(delta)
        except (TypeError, ValueError):
            raise TypeError(
                f"shift_phase at op[{idx}] has non-numeric phase value: "
                f"{delta!r} ({type(delta).__name__})")

        tone_vid = None
        for v in op.operands:
            if v.vtype == ValueType.TONE:
                tone_vid = v.vid
                break

        if tone_vid is not None:
            candidates.append((idx, tone_vid, delta_f))

    return candidates


def run_loop_strength_reduction(program: Program) -> Program:
    """Convert linear phase progressions to incremental updates.

    For each for-loop, find shift_phase ops with a constant delta applied
    every iteration. Mark them with metadata for downstream consumption.
    """
    result = clone_program(program)
    ops = result.ops
    loops = _find_loops(ops)
    inner_ranges = _inner_loop_ranges(loops)

    for loop_start, loop_end in reversed(loops):
        skip = inner_ranges.get((loop_start, loop_end), set())
        candidates = _detect_linear_phase_progression(ops, loop_start, loop_end,
                                                      skip)

        for op_idx, tone_vid, delta in candidates:
            op = ops[op_idx]
            new_attrs = dict(op.attrs)
            new_attrs["strength_reduced"] = True
            new_attrs["increment_delta"] = delta
            new_attrs["original_phase"] = delta
            new_attrs["incremental"] = True
            loop_attrs = ops[loop_start].attrs
            new_attrs["loop_count"] = loop_attrs.get("count",
                                                     loop_attrs.get("ub", 1))

            ops[op_idx] = Op(
                kind=op.kind,
                operands=op.operands,
                results=op.results,
                attrs=new_attrs,
            )

    result.ops = ops
    return result
