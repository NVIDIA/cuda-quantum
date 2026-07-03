# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verification passes for the pulse IR.

Checks linearity, monotone time, drive exclusivity, and cross-resonance
calibration heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    duration_of,
    is_linear_type,
    line_id_of,
    tone_id_of,
)

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


@dataclass()
class TypeCheckError:
    """Base class for all verification errors."""

    message: str
    op_index: int | None = None
    severity: str = "error"

    def __str__(self) -> str:
        loc = f" at op[{self.op_index}]" if self.op_index is not None else ""
        return f"[{self.severity}]{loc}: {self.message}"


@dataclass()
class LinearityViolation(TypeCheckError):
    """A linear value was produced more than once or consumed != 1 time.

    Severity is "warning" by default — list-based patterns in the Python
    frontend legitimately produce multi-consume SSA (sync over a list of
    lines).  The IR is still valid for lowering and scheduling.
    """

    value: Value | None = None
    detail: str = ""
    severity: str = "warning"


@dataclass()
class UnintentionalOverlapError(TypeCheckError):
    """Two drive ops on the same line overlap in time."""

    line_vid: int | None = None
    interval_a: tuple[float, float] = (0.0, 0.0)
    interval_b: tuple[float, float] = (0.0, 0.0)


@dataclass()
class BackwardTimeTravelError(TypeCheckError):
    """An operation references a time earlier than a predecessor on the same line."""

    line_vid: int | None = None
    expected_min: float = 0.0
    actual: float = 0.0


@dataclass()
class PhaseBookkeepingError(TypeCheckError):
    """Inconsistency detected in phase tracking for a tone."""

    tone_vid: int | None = None
    detail: str = ""


@dataclass()
class CrossResonanceMiscalibrationError(TypeCheckError):
    """Heuristic: possible cross-resonance tone mismatch."""

    target_qubit: int | None = None
    tone_tag: str = ""
    severity: str = "warning"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_linearity(program: Program) -> list[TypeCheckError]:
    """Every linear value must be produced exactly once and consumed exactly once."""
    errors: list[TypeCheckError] = []
    produced: dict[int, int] = {}  # vid -> op_index that produced it
    consumed: dict[int, int] = {}  # vid -> consume count

    for idx, op in enumerate(program.ops):
        for v in op.results:
            if not is_linear_type(v.vtype):
                continue
            if v.vid in produced:
                errors.append(
                    LinearityViolation(
                        message=
                        f"Linear value {v} produced again (first at op[{produced[v.vid]}])",
                        op_index=idx,
                        value=v,
                        detail="duplicate_production",
                    ))
            else:
                produced[v.vid] = idx

        for v in op.operands:
            if not is_linear_type(v.vtype):
                continue
            consumed[v.vid] = consumed.get(v.vid, 0) + 1

    for vid, prod_idx in produced.items():
        count = consumed.get(vid, 0)
        if count == 0:
            v_repr = f"%{vid}"
            errors.append(
                LinearityViolation(
                    message=
                    f"Linear value {v_repr} produced at op[{prod_idx}] but never consumed",
                    op_index=prod_idx,
                    detail="unconsumed",
                ))
        elif count > 1:
            v_repr = f"%{vid}"
            errors.append(
                LinearityViolation(
                    message=
                    f"Linear value {v_repr} consumed {count} times (expected 1)",
                    op_index=prod_idx,
                    detail="multiple_consumption",
                ))

    return errors


def check_monotone_time(program: Program) -> list[TypeCheckError]:
    """Verify that time only moves forward on each line."""
    errors: list[TypeCheckError] = []
    line_clocks: dict[int, float] = {}  # line_vid -> current time

    for idx, op in enumerate(program.ops):
        lid = line_id_of(op)
        if lid is None:
            continue

        start = float(op.attrs.get("start_vtu", line_clocks.get(lid, 0.0)))
        current = line_clocks.get(lid, 0.0)

        if start < current - 1e-12:
            errors.append(
                BackwardTimeTravelError(
                    message=
                    f"Time goes backward on line %{lid}: expected >= {current:.4f}, got {start:.4f}",
                    op_index=idx,
                    line_vid=lid,
                    expected_min=current,
                    actual=start,
                ))

        dur = duration_of(op)
        end = start + dur
        if end > current:
            line_clocks[lid] = end

        for r in op.results:
            if r.vtype in (ValueType.DRIVE_LINE, ValueType.READOUT_LINE):
                line_clocks[r.vid] = line_clocks.get(lid, 0.0)

    return errors


def check_drive_exclusivity(program: Program) -> list[TypeCheckError]:
    """Drive ops on the same line lineage must be totally ordered, non-overlapping.

    Only meaningful after scheduling — if no ops carry ``start_vtu`` attrs
    the check is skipped (unscheduled programs trivially alias at t=0).
    """
    errors: list[TypeCheckError] = []

    has_schedule = any("start_vtu" in op.attrs
                       for op in program.ops
                       if op.kind == OpKind.DRIVE)
    if not has_schedule:
        return errors

    line_intervals: dict[int, list[tuple[float, float, int]]] = {}

    for idx, op in enumerate(program.ops):
        if op.kind != OpKind.DRIVE:
            continue
        lid = line_id_of(op)
        if lid is None:
            continue

        start = float(op.attrs.get("start_vtu", 0.0))
        dur = duration_of(op)
        end = start + dur

        if lid not in line_intervals:
            line_intervals[lid] = []

        for prev_start, prev_end, prev_idx in line_intervals[lid]:
            if start < prev_end - 1e-12 and end > prev_start + 1e-12:
                errors.append(
                    UnintentionalOverlapError(
                        message=
                        (f"Drive ops overlap on line %{lid}: "
                         f"op[{prev_idx}] [{prev_start:.2f},{prev_end:.2f}) vs "
                         f"op[{idx}] [{start:.2f},{end:.2f})"),
                        op_index=idx,
                        line_vid=lid,
                        interval_a=(prev_start, prev_end),
                        interval_b=(start, end),
                    ))

        line_intervals[lid].append((start, end, idx))

    return errors


def check_cr_miscalibration(program: Program) -> list[TypeCheckError]:
    """Heuristic check for cross-resonance tone tag mismatches.

    If a drive op is tagged as cross-resonance (attrs['cr_target'] is set),
    verify the tone tag matches the target qubit's frequency label.
    """
    errors: list[TypeCheckError] = []

    for idx, op in enumerate(program.ops):
        if op.kind != OpKind.DRIVE:
            continue
        cr_target = op.attrs.get("cr_target")
        if cr_target is None:
            continue

        tone_tag = op.attrs.get("tone_tag", "")
        expected_tag = f"q{cr_target}"

        if tone_tag and expected_tag not in tone_tag:
            errors.append(
                CrossResonanceMiscalibrationError(
                    message=
                    (f"Cross-resonance drive at op[{idx}] targets qubit {cr_target} "
                     f"but tone tag is '{tone_tag}' (expected to contain '{expected_tag}')"
                    ),
                    op_index=idx,
                    target_qubit=cr_target,
                    tone_tag=tone_tag,
                ))

    return errors


def check_loop_structure(program: Program) -> list[TypeCheckError]:
    """Verify FOR_LOOP/END_FOR are balanced and carry required attrs."""
    errors: list[TypeCheckError] = []
    stack: list[int] = []

    for idx, op in enumerate(program.ops):
        if op.kind == OpKind.FOR_LOOP:
            stack.append(idx)
            if "ub" not in op.attrs and "count" not in op.attrs:
                errors.append(
                    TypeCheckError(
                        message=
                        f"FOR_LOOP at op[{idx}] missing 'ub' or 'count' attr",
                        op_index=idx,
                    ))
        elif op.kind == OpKind.END_FOR:
            if not stack:
                errors.append(
                    TypeCheckError(
                        message=
                        f"END_FOR at op[{idx}] without matching FOR_LOOP",
                        op_index=idx,
                    ))
            else:
                stack.pop()

    for start_idx in stack:
        errors.append(
            TypeCheckError(
                message=f"FOR_LOOP at op[{start_idx}] without matching END_FOR",
                op_index=start_idx,
            ))

    return errors


def check_waveform_validity(program: Program) -> list[TypeCheckError]:
    """Check waveform construction attrs for basic validity."""
    errors: list[TypeCheckError] = []

    for idx, op in enumerate(program.ops):
        if op.kind != OpKind.MAKE_WAVEFORM:
            continue

        dur = op.attrs.get("duration_vtu")
        if dur is not None and float(dur) <= 0:
            errors.append(
                TypeCheckError(
                    message=
                    f"Waveform at op[{idx}] has non-positive duration: {dur}",
                    op_index=idx,
                ))

        amp = op.attrs.get("amplitude")
        if amp is not None:
            amp_vals = amp if isinstance(amp, (list, tuple)) else [amp]
            for av in amp_vals:
                try:
                    a = abs(complex(av)) if isinstance(av,
                                                       complex) else float(av)
                    if not (-1e6 < a < 1e6):
                        errors.append(
                            TypeCheckError(
                                message=
                                f"Waveform at op[{idx}] has extreme amplitude: {a}",
                                op_index=idx,
                                severity="warning",
                            ))
                except (TypeError, ValueError):
                    errors.append(
                        TypeCheckError(
                            message=
                            f"Waveform at op[{idx}] has non-numeric amplitude component: {av!r}",
                            op_index=idx,
                        ))

        sigma = op.attrs.get("sigma")
        if sigma is not None and float(sigma) <= 0:
            errors.append(
                TypeCheckError(
                    message=
                    f"Waveform at op[{idx}] has non-positive sigma: {sigma}",
                    op_index=idx,
                ))

    return errors


# ---------------------------------------------------------------------------
# Aggregate verifier
# ---------------------------------------------------------------------------


def verify(program: Program, *, strict: bool = False) -> list[TypeCheckError]:
    """Run all verification checks and return collected errors/warnings.

    Parameters
    ----------
    strict : bool
        If True, raise ``RuntimeError`` on the first error-severity issue.
    """
    errors: list[TypeCheckError] = []
    errors.extend(check_linearity(program))
    errors.extend(check_loop_structure(program))
    errors.extend(check_waveform_validity(program))
    errors.extend(check_monotone_time(program))
    errors.extend(check_drive_exclusivity(program))
    errors.extend(check_cr_miscalibration(program))

    if strict:
        for e in errors:
            if e.severity == "error":
                raise RuntimeError(str(e))

    return errors
