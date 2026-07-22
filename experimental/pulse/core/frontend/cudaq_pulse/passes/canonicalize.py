# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
    line_id_of,
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
    line_clocks: dict[int, float] = {}
    result: list[Op] = []

    for op in ops:
        if is_loop_or_barrier(op) and op.kind not in (OpKind.SYNC,):
            line_clocks.clear()
            result.append(op)
        elif op.kind == OpKind.SYNC:
            sync_vids = [
                v.vid
                for v in op.operands
                if v.vtype in (ValueType.DRIVE_LINE, ValueType.READOUT_LINE)
            ]
            if sync_vids:
                times = [line_clocks.get(vid, 0.0) for vid in sync_vids]
                if len(set(round(t, 10) for t in times)) <= 1:
                    continue
                sync_time = max(times)
                for vid in sync_vids:
                    line_clocks[vid] = sync_time
            result.append(op)
        else:
            lid = line_id_of(op)
            if lid is not None:
                dur = duration_of(op)
                line_clocks[lid] = line_clocks.get(lid, 0.0) + dur
            result.append(op)

    return result


def _dead_line_elim(ops: list[Op]) -> list[Op]:
    """Remove lines that are allocated but never driven or read."""
    used_lines: set[int] = set()
    alloc_indices: dict[int, int] = {}

    for idx, op in enumerate(ops):
        if op.kind in (OpKind.ALLOC_DRIVE, OpKind.ALLOC_READOUT):
            for v in op.results:
                if v.vtype in (ValueType.DRIVE_LINE, ValueType.READOUT_LINE):
                    alloc_indices[v.vid] = idx
        elif op.kind in (OpKind.DRIVE, OpKind.READOUT, OpKind.SHIFT_PHASE,
                         OpKind.SET_PHASE, OpKind.WAIT):
            lid = line_id_of(op)
            if lid is not None:
                used_lines.add(lid)

    dead_vids = set(alloc_indices.keys()) - used_lines
    if not dead_vids:
        return ops

    dead_indices = {alloc_indices[vid] for vid in dead_vids}
    return [op for idx, op in enumerate(ops) if idx not in dead_indices]


def _idle_compression(ops: list[Op]) -> list[Op]:
    """Merge adjacent waits on the same line."""
    result: list[Op] = []

    for op in ops:
        if op.kind == OpKind.WAIT and result:
            prev = result[-1]
            if prev.kind == OpKind.WAIT and line_id_of(prev) == line_id_of(op):
                merged_dur = duration_of(prev) + duration_of(op)
                merged_attrs = dict(prev.attrs)
                merged_attrs["duration_vtu"] = merged_dur
                result[-1] = Op(
                    kind=OpKind.WAIT,
                    operands=prev.operands,
                    results=prev.results,
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
