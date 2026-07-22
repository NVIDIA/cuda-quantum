# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fluent ProgramBuilder for constructing pulse Programs programmatically.

Provides a builder API that mirrors the paper's ``pulse_ref`` style:
``get_drive_line``, ``drive``, ``gaussian``, ``drag``, ``square``, ``wait``,
``sync``, ``shift_phase``, ``set_phase``. All methods return updated line/tone
handles for linear chaining.
"""

from __future__ import annotations

import math
from typing import Any

from .ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    _mk,
    _reset_vid_counter,
)


class ProgramBuilder:
    """Fluent builder that constructs a ``Program`` incrementally.

    Usage::

        p = ProgramBuilder("bell", clock_ghz=2.0)
        d0, t0 = p.get_drive_line(0, 5.0e9)
        sx = p.drag(40, 0.25, 10.0, 0.5)
        d0, t0 = p.drive(d0, sx, t0)
        program = p.build()   # returns the underlying Program
    """

    def __init__(self, name: str, clock_ghz: float = 2.0):
        _reset_vid_counter(0)
        self._name = name
        self._clock_ghz = clock_ghz
        self._ops: list[Op] = []
        self._values: list[Value] = []
        self._qubit_freq_hz: dict[int, float] = {}

    def build(self) -> Program:
        """Finalize and return the underlying ``Program``."""
        return Program(
            name=self._name,
            clock_ghz=self._clock_ghz,
            ops=list(self._ops),
            values=list(self._values),
            qubit_freq_hz=dict(self._qubit_freq_hz),
        )

    def get_drive_line(self, qubit: int, freq_hz: float) -> tuple[Value, Value]:
        """Allocate a drive line and tone for the given qubit."""
        self._qubit_freq_hz[qubit] = freq_hz
        d = _mk(ValueType.DRIVE_LINE, f"d{qubit}")
        t = _mk(ValueType.TONE, f"t{qubit}")
        self._values.extend([d, t])
        self._ops.append(
            Op(
                kind=OpKind.ALLOC_DRIVE,
                operands=(),
                results=(d, t),
                attrs={
                    "qubit": qubit,
                    "frequency_hz": freq_hz
                },
            ))
        return d, t

    def get_readout_line(self, qubit: int,
                         freq_hz: float) -> tuple[Value, Value]:
        """Allocate a readout line and tone."""
        r = _mk(ValueType.READOUT_LINE, f"ro{qubit}")
        t = _mk(ValueType.TONE, f"rot{qubit}")
        self._values.extend([r, t])
        self._ops.append(
            Op(
                kind=OpKind.ALLOC_READOUT,
                operands=(),
                results=(r, t),
                attrs={
                    "qubit": qubit,
                    "frequency_hz": freq_hz
                },
            ))
        return r, t

    def drive(self, line: Value, waveform: Value,
              tone: Value) -> tuple[Value, Value]:
        """Emit a drive operation. Returns updated (line, tone)."""
        dur = 0.0
        for op in reversed(self._ops):
            if op.results and any(r.vid == waveform.vid for r in op.results):
                dur = float(op.attrs.get("duration_vtu", 0.0))
                break

        new_line = _mk(line.vtype, line.name)
        new_tone = _mk(tone.vtype, tone.name)
        self._values.extend([new_line, new_tone])

        self._ops.append(
            Op(
                kind=OpKind.DRIVE,
                operands=(line, waveform, tone),
                results=(new_line, new_tone),
                attrs={"duration_vtu": dur},
            ))
        return new_line, new_tone

    def wait(self, line: Value, duration_vtu: float) -> Value:
        """Insert an idle wait on a drive line."""
        new_line = _mk(line.vtype, line.name)
        self._values.append(new_line)
        self._ops.append(
            Op(
                kind=OpKind.WAIT,
                operands=(line,),
                results=(new_line,),
                attrs={"duration_vtu": duration_vtu},
            ))
        return new_line

    def sync(self, *lines: Value) -> tuple[Value, ...]:
        """Synchronize multiple lines. Returns updated line handles."""
        new_lines = tuple(_mk(l.vtype, l.name) for l in lines)
        self._values.extend(new_lines)
        self._ops.append(
            Op(
                kind=OpKind.SYNC,
                operands=lines,
                results=new_lines,
                attrs={},
            ))
        return new_lines

    def shift_phase(self, tone: Value, delta_rad: float) -> Value:
        """Shift the phase of a tone."""
        new_tone = _mk(tone.vtype, tone.name)
        self._values.append(new_tone)
        self._ops.append(
            Op(
                kind=OpKind.SHIFT_PHASE,
                operands=(tone,),
                results=(new_tone,),
                attrs={"delta_rad": delta_rad},
            ))
        return new_tone

    def set_phase(self, tone: Value, phase_rad: float) -> Value:
        """Set the absolute phase of a tone."""
        new_tone = _mk(tone.vtype, tone.name)
        self._values.append(new_tone)
        self._ops.append(
            Op(
                kind=OpKind.SET_PHASE,
                operands=(tone,),
                results=(new_tone,),
                attrs={"phase_rad": phase_rad},
            ))
        return new_tone

    # -- Waveform constructors --

    def gaussian(self, duration_vtu: float, amplitude: float,
                 sigma: float) -> Value:
        """Create a Gaussian waveform."""
        w = _mk(ValueType.WAVEFORM, "gaussian")
        self._values.append(w)
        self._ops.append(
            Op(
                kind=OpKind.MAKE_WAVEFORM,
                operands=(),
                results=(w,),
                attrs={
                    "waveform_type": "gaussian",
                    "duration_vtu": duration_vtu,
                    "amplitude": amplitude,
                    "sigma": sigma,
                },
            ))
        return w

    def drag(self, duration_vtu: float, amplitude: float, sigma: float,
             beta: float) -> Value:
        """Create a DRAG waveform."""
        w = _mk(ValueType.WAVEFORM, "drag")
        self._values.append(w)
        self._ops.append(
            Op(
                kind=OpKind.MAKE_WAVEFORM,
                operands=(),
                results=(w,),
                attrs={
                    "waveform_type": "drag",
                    "duration_vtu": duration_vtu,
                    "amplitude": amplitude,
                    "sigma": sigma,
                    "beta": beta,
                },
            ))
        return w

    def square(self, duration_vtu: float, amplitude: complex) -> Value:
        """Create a constant (square) waveform."""
        w = _mk(ValueType.WAVEFORM, "square")
        self._values.append(w)
        self._ops.append(
            Op(
                kind=OpKind.MAKE_WAVEFORM,
                operands=(),
                results=(w,),
                attrs={
                    "waveform_type":
                        "square",
                    "duration_vtu":
                        duration_vtu,
                    "amplitude":
                        abs(amplitude),
                    "phase":
                        math.atan2(amplitude.imag, amplitude.real)
                        if isinstance(amplitude, complex) else 0.0,
                },
            ))
        return w

    def cosine(self, duration_vtu: float, amplitude: float) -> Value:
        """Create a cosine waveform."""
        w = _mk(ValueType.WAVEFORM, "cosine")
        self._values.append(w)
        self._ops.append(
            Op(
                kind=OpKind.MAKE_WAVEFORM,
                operands=(),
                results=(w,),
                attrs={
                    "waveform_type": "cosine",
                    "duration_vtu": duration_vtu,
                    "amplitude": amplitude,
                },
            ))
        return w

    def gaussian_square(self, duration_vtu: float, amplitude: float,
                        sigma: float, flat_top_vtu: float) -> Value:
        """Create a Gaussian-square (flat-top Gaussian) waveform."""
        w = _mk(ValueType.WAVEFORM, "gaussian_square")
        self._values.append(w)
        self._ops.append(
            Op(
                kind=OpKind.MAKE_WAVEFORM,
                operands=(),
                results=(w,),
                attrs={
                    "waveform_type": "gaussian_square",
                    "duration_vtu": duration_vtu,
                    "amplitude": amplitude,
                    "sigma": sigma,
                    "flat_top_vtu": flat_top_vtu,
                },
            ))
        return w

    def readout(self, line: Value, waveform: Value,
                tone: Value) -> tuple[Value, Value]:
        """Emit a readout operation. Returns updated (line, tone)."""
        new_line = _mk(line.vtype, line.name)
        new_tone = _mk(tone.vtype, tone.name)
        self._values.extend([new_line, new_tone])
        self._ops.append(
            Op(
                kind=OpKind.READOUT,
                operands=(line, waveform, tone),
                results=(new_line, new_tone),
                attrs={},
            ))
        return new_line, new_tone
