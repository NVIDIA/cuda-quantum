# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed-buffer encoder for zero-copy Program → MLIR module construction.

Encodes a ``Program`` as a flat ``numpy.ndarray[int64]`` that the C++
``PulseModuleBuilder.build_from_packed()`` consumes via raw pointer
iteration — one FFI crossing for the entire program.

Wire format
-----------
Each op is a variable-length record of int64 words::

    [header] [payload_0] [payload_1] ...

Header layout (64 bits):
    bits  0-7 : OpCode  (uint8 enum)
    bits  8-15: payload length N (uint8, count of following int64 words)
    bits 16-63: reserved (zero)

Floats are stored as their IEEE-754 double bit-pattern reinterpreted
as int64 (``struct.pack('d', x)`` → ``struct.unpack('q', …)``).
"""
from __future__ import annotations

import struct
from typing import Any

import numpy as np

from .passes.ir_types import OpKind, Program, ValueType

# ── OpCode enum (must match C++ kOp* constants in bindings.cpp) ──────────

ALLOC_DRIVE = 0
ALLOC_READOUT = 1
ALLOC_TONE = 2
WF_GAUSSIAN = 3
WF_SQUARE = 4
WF_DRAG = 5
WF_COSINE = 6
WF_TANH_RAMP = 7
WF_GAUSS_SQUARE = 8
WF_CUSTOM = 9
DRIVE = 10
READOUT = 11
SYNC = 12
WAIT = 13
SHIFT_PHASE = 14
SET_PHASE = 15
SHIFT_FREQ = 16
SET_FREQ = 17

_UNSCHEDULED = -1

_WF_TYPE_MAP = {
    "gaussian": WF_GAUSSIAN,
    "square": WF_SQUARE,
    "drag": WF_DRAG,
    "cosine": WF_COSINE,
    "tanh_ramp": WF_TANH_RAMP,
    "gaussian_square": WF_GAUSS_SQUARE,
}

_VTYPE_INT = {
    ValueType.DRIVE_LINE: 0,
    ValueType.READOUT_LINE: 1,
    ValueType.TONE: 2,
}

_pack_d = struct.Struct("=d")
_unpack_q = struct.Struct("=q")


def _f2i(x: float) -> int:
    """Bit-cast a float64 to int64."""
    return _unpack_q.unpack(_pack_d.pack(float(x)))[0]


def _header(opcode: int, payload_len: int) -> int:
    return opcode | (payload_len << 8)


def pack_program(prog: Program) -> np.ndarray:
    """Encode a Program into a flat int64 numpy array (zero-copy ready)."""
    buf = np.empty(len(prog.ops) * 10, dtype=np.int64)
    c = 0

    for op in prog.ops:
        kind = op.kind
        a = op.attrs

        if kind == OpKind.ALLOC_DRIVE:
            qubit = int(a["qubit"])
            lv = op.results[0].vid
            tv = op.results[1].vid
            buf[c] = _header(ALLOC_DRIVE, 3)
            buf[c + 1] = qubit
            buf[c + 2] = lv
            buf[c + 3] = tv
            c += 4

        elif kind == OpKind.ALLOC_READOUT:
            qubit = int(a["qubit"])
            lv = op.results[0].vid
            tv = op.results[1].vid
            buf[c] = _header(ALLOC_READOUT, 3)
            buf[c + 1] = qubit
            buf[c + 2] = lv
            buf[c + 3] = tv
            c += 4

        elif kind == OpKind.ALLOC_TONE:
            tv = op.results[0].vid
            freq = float(a.get("frequency_hz", 0.0))
            phase = float(a.get("phase_rad", 0.0))
            buf[c] = _header(ALLOC_TONE, 3)
            buf[c + 1] = tv
            buf[c + 2] = _f2i(freq)
            buf[c + 3] = _f2i(phase)
            c += 4

        elif kind == OpKind.MAKE_WAVEFORM:
            rv = op.results[0].vid
            wf_type = a.get("waveform_type", "")
            dur = int(a.get("duration_vtu", 0))
            opcode = _WF_TYPE_MAP.get(wf_type, WF_CUSTOM)

            if opcode == WF_GAUSSIAN:
                amp = _extract_real(a.get("amplitude", 0.0))
                sigma = float(a.get("sigma", 1.0))
                buf[c] = _header(WF_GAUSSIAN, 4)
                buf[c + 1] = rv
                buf[c + 2] = dur
                buf[c + 3] = _f2i(amp)
                buf[c + 4] = _f2i(sigma)
                c += 5

            elif opcode == WF_SQUARE:
                re, im = _extract_complex(a.get("amplitude", 0.0))
                buf[c] = _header(WF_SQUARE, 4)
                buf[c + 1] = rv
                buf[c + 2] = dur
                buf[c + 3] = _f2i(re)
                buf[c + 4] = _f2i(im)
                c += 5

            elif opcode == WF_DRAG:
                amp = _extract_real(a.get("amplitude", 0.0))
                sigma = float(a.get("sigma", 1.0))
                beta = float(a.get("beta", 0.0))
                buf[c] = _header(WF_DRAG, 5)
                buf[c + 1] = rv
                buf[c + 2] = dur
                buf[c + 3] = _f2i(amp)
                buf[c + 4] = _f2i(sigma)
                buf[c + 5] = _f2i(beta)
                c += 6

            elif opcode == WF_COSINE:
                amp = _extract_real(a.get("amplitude", 0.0))
                buf[c] = _header(WF_COSINE, 3)
                buf[c + 1] = rv
                buf[c + 2] = dur
                buf[c + 3] = _f2i(amp)
                c += 4

            elif opcode == WF_TANH_RAMP:
                amp = _extract_real(a.get("amplitude", 0.0))
                sigma = float(a.get("sigma", 1.0))
                buf[c] = _header(WF_TANH_RAMP, 4)
                buf[c + 1] = rv
                buf[c + 2] = dur
                buf[c + 3] = _f2i(amp)
                buf[c + 4] = _f2i(sigma)
                c += 5

            elif opcode == WF_GAUSS_SQUARE:
                amp = _extract_real(a.get("amplitude", 0.0))
                sigma = float(a.get("sigma", 1.0))
                risefall = int(a.get("risefall", 0))
                buf[c] = _header(WF_GAUSS_SQUARE, 5)
                buf[c + 1] = rv
                buf[c + 2] = dur
                buf[c + 3] = _f2i(amp)
                buf[c + 4] = _f2i(sigma)
                buf[c + 5] = risefall
                c += 6

            else:
                buf[c] = _header(WF_CUSTOM, 2)
                buf[c + 1] = rv
                buf[c + 2] = dur
                c += 3

        elif kind == OpKind.DRIVE:
            lv = op.operands[0].vid
            wv = op.operands[1].vid
            tv = op.operands[2].vid
            rlv = op.results[0].vid
            rtv = op.results[1].vid
            sv = int(a["start_vtu"]) if "start_vtu" in a else _UNSCHEDULED
            dv = int(a["duration_vtu"]) if "duration_vtu" in a else _UNSCHEDULED
            buf[c] = _header(DRIVE, 7)
            buf[c + 1] = lv
            buf[c + 2] = wv
            buf[c + 3] = tv
            buf[c + 4] = rlv
            buf[c + 5] = rtv
            buf[c + 6] = sv
            buf[c + 7] = dv
            c += 8

        elif kind == OpKind.READOUT:
            lv = op.operands[0].vid
            wv = op.operands[1].vid
            tv = op.operands[2].vid
            rlv = op.results[0].vid
            rtv = op.results[1].vid
            mv = op.results[2].vid
            buf[c] = _header(READOUT, 6)
            buf[c + 1] = lv
            buf[c + 2] = wv
            buf[c + 3] = tv
            buf[c + 4] = rlv
            buf[c + 5] = rtv
            buf[c + 6] = mv
            c += 7

        elif kind == OpKind.SYNC:
            n = len(op.operands)
            payload_len = 1 + 3 * n
            buf[c] = _header(SYNC, payload_len)
            buf[c + 1] = n
            for j in range(n):
                in_vid = op.operands[j].vid
                out_vid = op.results[j].vid if j < len(op.results) else in_vid
                vtype = _VTYPE_INT.get(op.results[j].vtype, 0) if j < len(
                    op.results) else 0
                buf[c + 2 + 3 * j] = in_vid
                buf[c + 3 + 3 * j] = out_vid
                buf[c + 4 + 3 * j] = vtype
            c += 1 + payload_len

        elif kind == OpKind.WAIT:
            lv = op.operands[0].vid
            rlv = op.results[0].vid
            dv = int(a.get("duration_vtu", 0))
            buf[c] = _header(WAIT, 3)
            buf[c + 1] = lv
            buf[c + 2] = rlv
            buf[c + 3] = dv
            c += 4

        elif kind == OpKind.SHIFT_PHASE:
            tv = op.operands[0].vid
            rtv = op.results[0].vid if op.results else tv
            delta = float(a.get("delta_rad", a.get("delta", 0.0)))
            buf[c] = _header(SHIFT_PHASE, 3)
            buf[c + 1] = tv
            buf[c + 2] = rtv
            buf[c + 3] = _f2i(delta)
            c += 4

        elif kind == OpKind.SET_PHASE:
            tv = op.operands[0].vid
            rtv = op.results[0].vid if op.results else tv
            phase = float(a.get("phase_rad", 0.0))
            buf[c] = _header(SET_PHASE, 3)
            buf[c + 1] = tv
            buf[c + 2] = rtv
            buf[c + 3] = _f2i(phase)
            c += 4

        # for_loop / end_for are structural, not encoded
        # (scheduling is done in Python before packing)

    return buf[:c]


def _extract_real(val: Any) -> float:
    if isinstance(val, complex):
        return val.real
    return float(val)


def _extract_complex(val: Any) -> tuple[float, float]:
    if isinstance(val, complex):
        return (val.real, val.imag)
    return (float(val), 0.0)


def emit_pulse_module_packed(prog: Program) -> Any:
    """Build an in-memory PulseModule via the packed-buffer zero-copy path.

    Returns a ``PulseModule`` whose ``.print()`` gives MLIR text and
    ``.run_passes()`` / ``.run_full_lowering()`` operate in-memory.
    """
    from _cudaq_pulse_native import PulseModuleBuilder

    buf = pack_program(prog)
    n_qubits = len(prog.qubit_freq_hz)
    freq_arr = np.zeros(n_qubits, dtype=np.float64)
    for q, f in prog.qubit_freq_hz.items():
        if q < n_qubits:
            freq_arr[q] = f
    builder = PulseModuleBuilder()
    return builder.build_from_packed(buf, prog.clock_ghz, n_qubits, freq_arr)
