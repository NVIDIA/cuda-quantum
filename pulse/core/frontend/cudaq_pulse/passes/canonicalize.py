# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Canonicalization pass for the pulse IR.

Attempts to call the native CAPI (cudaqPulseRunCanonicalize); falls back to a
pure-Python implementation of the same transforms.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
from typing import Any

from .ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    clone_program,
    duration_of,
    is_loop_or_barrier,
    waveform_of,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native CAPI attempt
# ---------------------------------------------------------------------------

_native_lib = None


def _try_load_native() -> Any | None:
    """Attempt to load the cudaq-pulse native library."""
    global _native_lib
    if _native_lib is not None:
        return _native_lib

    for name in ("libcudaq_pulse", "cudaq_pulse"):
        path = ctypes.util.find_library(name)
        if path:
            try:
                _native_lib = ctypes.CDLL(path)
                return _native_lib
            except OSError:
                continue
    return None


# ---------------------------------------------------------------------------
# Pure-Python canonicalization transforms
# ---------------------------------------------------------------------------


def _redundant_sync_elim(ops: list[Op]) -> list[Op]:
    """Remove syncs where all input lines already share the same time."""
    line_roots: dict[int, int] = {}
    line_clocks: dict[int, float] = {}
    replacements: dict[int, Value] = {}
    result: list[Op] = []

    def resolve(value: Value) -> Value:
        seen: set[int] = set()
        while value.vid in replacements and value.vid not in seen:
            seen.add(value.vid)
            value = replacements[value.vid]
        return value

    def line_values(values: tuple[Value, ...]) -> list[Value]:
        return [
            value for value in values if value.vtype in (
                ValueType.DRIVE_LINE,
                ValueType.READOUT_LINE,
            )
        ]

    for original in ops:
        operands = tuple(resolve(value) for value in original.operands)
        op = Op(original.kind, operands, original.results, dict(original.attrs))
        inputs = line_values(operands)
        outputs = line_values(op.results)

        if op.kind in (OpKind.ALLOC_DRIVE, OpKind.ALLOC_READOUT):
            for output in outputs:
                line_roots[output.vid] = output.vid
                line_clocks[output.vid] = 0.0
            result.append(op)
            continue

        if is_loop_or_barrier(op) and op.kind not in (OpKind.SYNC,):
            line_clocks.clear()
            result.append(op)
        elif op.kind == OpKind.SYNC:
            roots = [line_roots.get(value.vid, value.vid) for value in inputs]
            if roots:
                times = [line_clocks.get(root, 0.0) for root in roots]
                if len(set(round(t, 10) for t in times)) <= 1:
                    for output, source in zip(outputs, inputs):
                        replacements[output.vid] = source
                    continue
                sync_time = max(times)
                for root in roots:
                    line_clocks[root] = sync_time
                for output, root in zip(outputs, roots):
                    line_roots[output.vid] = root
            result.append(op)
        else:
            roots = [line_roots.get(value.vid, value.vid) for value in inputs]
            for output, root in zip(outputs, roots):
                line_roots[output.vid] = root
            duration = duration_of(op)
            for root in set(roots):
                line_clocks[root] = line_clocks.get(root, 0.0) + duration
            result.append(op)

    return result


def _dead_line_elim(ops: list[Op]) -> list[Op]:
    """Remove lines that are allocated but never driven or read."""
    linear_types = (ValueType.DRIVE_LINE, ValueType.READOUT_LINE,
                    ValueType.TONE)
    roots: dict[int, int] = {}
    used_roots: set[int] = set()
    alloc_roots: dict[int, set[int]] = {}

    for idx, op in enumerate(ops):
        if op.kind in (OpKind.ALLOC_DRIVE, OpKind.ALLOC_READOUT):
            allocation = set()
            for value in op.results:
                if value.vtype in linear_types:
                    roots[value.vid] = value.vid
                    allocation.add(value.vid)
            alloc_roots[idx] = allocation
            continue

        inputs_by_type: dict[ValueType, list[int]] = {}
        for value in op.operands:
            if value.vtype not in linear_types:
                continue
            root = roots.get(value.vid, value.vid)
            used_roots.add(root)
            inputs_by_type.setdefault(value.vtype, []).append(root)

        output_offsets: dict[ValueType, int] = {}
        for value in op.results:
            candidates = inputs_by_type.get(value.vtype, [])
            offset = output_offsets.get(value.vtype, 0)
            if offset < len(candidates):
                roots[value.vid] = candidates[offset]
                output_offsets[value.vtype] = offset + 1
            elif value.vtype in linear_types:
                roots[value.vid] = value.vid

    dead_indices = {
        index for index, allocation in alloc_roots.items()
        if allocation.isdisjoint(used_roots)
    }
    if not dead_indices:
        return ops
    return [op for idx, op in enumerate(ops) if idx not in dead_indices]


def _idle_compression(ops: list[Op]) -> list[Op]:
    """Merge adjacent waits on the same line."""
    result: list[Op] = []

    for op in ops:
        if op.kind == OpKind.WAIT and result:
            prev = result[-1]
            previous_lines = [
                value for value in prev.results if value.vtype in (
                    ValueType.DRIVE_LINE,
                    ValueType.READOUT_LINE,
                )
            ]
            current_lines = [
                value for value in op.operands if value.vtype in (
                    ValueType.DRIVE_LINE,
                    ValueType.READOUT_LINE,
                )
            ]
            if (prev.kind == OpKind.WAIT and previous_lines and
                    current_lines and
                    previous_lines[0].vid == current_lines[0].vid):
                merged_dur = duration_of(prev) + duration_of(op)
                merged_attrs = dict(prev.attrs)
                merged_attrs["duration_vtu"] = merged_dur
                result[-1] = Op(
                    kind=OpKind.WAIT,
                    operands=prev.operands,
                    results=op.results,
                    attrs=merged_attrs,
                )
                continue
        result.append(op)

    return result


def _waveform_cse(ops: list[Op]) -> list[Op]:
    """Deduplicate identical waveform constructions within the same scope."""
    seen: dict[tuple, Value] = {}
    replacements: dict[int, Value] = {}
    result: list[Op] = []

    for op in ops:
        if is_loop_or_barrier(op):
            seen.clear()
        if op.kind == OpKind.MAKE_WAVEFORM:
            key = (
                op.attrs.get("waveform_type"),
                op.attrs.get("duration_vtu"),
                op.attrs.get("amplitude"),
                op.attrs.get("frequency"),
                op.attrs.get("phase"),
            )
            if key in seen and op.results:
                replacements[op.results[0].vid] = seen[key]
                continue
            elif op.results:
                seen[key] = op.results[0]

        new_operands = tuple(replacements.get(v.vid, v) for v in op.operands)
        result.append(
            Op(
                kind=op.kind,
                operands=new_operands,
                results=op.results,
                attrs=op.attrs,
            ))

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_canonicalize(program: Program) -> Program:
    """Run canonicalization passes on the program.

    Runs pure-Python implementations of redundant-sync elimination,
    dead-line elimination, idle compression, and waveform CSE.
    """
    result = clone_program(program)
    result.ops = _redundant_sync_elim(result.ops)
    result.ops = _dead_line_elim(result.ops)
    result.ops = _idle_compression(result.ops)
    result.ops = _waveform_cse(result.ops)
    return result
