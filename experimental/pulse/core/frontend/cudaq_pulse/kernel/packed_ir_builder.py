# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PackedIRBuilder -- fused tracer-to-packed-buffer emitter.

Replaces PythonIRBuilder + lower.py + pack_program on the fast path.
Writes directly into a flat numpy int64 array during kernel tracing,
producing a buffer that ``PulseModuleBuilder.build_from_packed()``
consumes via zero-copy FFI.
"""
from __future__ import annotations

import struct
from typing import Any

import numpy as np

from .ir_builder import IRValue, Parameter, CompilationError

_pack_d = struct.Struct("=d")
_unpack_q = struct.Struct("=q")

# OpCodes -- must match packed_emit.py and bindings.cpp kOp* constants
_ALLOC_DRIVE = 0
_ALLOC_READOUT = 1
_ALLOC_TONE = 2
_WF_GAUSSIAN = 3
_WF_SQUARE = 4
_WF_DRAG = 5
_WF_COSINE = 6
_WF_TANH_RAMP = 7
_WF_GAUSS_SQUARE = 8
_WF_CUSTOM = 9
_DRIVE = 10
_READOUT = 11
_SYNC = 12
_WAIT = 13
_SHIFT_PHASE = 14
_SET_PHASE = 15
_SHIFT_FREQ = 16
_SET_FREQ = 17
_PARAM = 18

_WF_MAP = {
    "gaussian": _WF_GAUSSIAN,
    "square": _WF_SQUARE,
    "drag": _WF_DRAG,
    "cosine": _WF_COSINE,
    "tanh_ramp": _WF_TANH_RAMP,
    "gaussian_square": _WF_GAUSS_SQUARE,
}

_UNSCHEDULED = -1


def _f2i(x: float) -> int:
    return _unpack_q.unpack(_pack_d.pack(float(x)))[0]


def _header(opcode: int, payload_len: int, param_mask: int = 0) -> int:
    return opcode | (payload_len << 8) | (param_mask << 16)


def _real(val: Any) -> float:
    return val.real if isinstance(val, complex) else float(val)


def _complex_parts(val: Any) -> tuple[float, float]:
    if isinstance(val, complex):
        return (val.real, val.imag)
    return (float(val), 0.0)


class PackedIRBuilder:
    """Fused trace-to-buffer builder.

    Implements the same ``emit()`` interface as ``PythonIRBuilder`` so it
    can be dropped into the ``PulseIREmitter`` unchanged.  Instead of
    accumulating Python objects, it writes packed int64 records into a
    pre-allocated numpy buffer.
    """

    def __init__(
        self,
        name: str = "main",
        *,
        clock_ghz: float = 2.0,
        qubit_freq_hz: dict[int, float] | None = None,
    ):
        self.name = name
        self.clock_ghz = clock_ghz
        self._freq_hz = qubit_freq_hz or {}
        self._buf = np.empty(8192, dtype=np.int64)
        self._cur = 0
        self._next_id = 0
        self._next_qubit = 0
        self._qref_to_qubit: dict[int, int] = {}
        self._wf_attrs: dict[int, dict[str, Any]] = {}
        self._param_vids: dict[int, int] = {}  # Parameter.index → vid

    def _mk(self, vtype: str, name: str = "") -> IRValue:
        v = IRValue(self._next_id, vtype, name)
        self._next_id += 1
        return v

    def _ensure(self, n: int) -> None:
        while self._cur + n >= len(self._buf):
            self._buf = np.resize(self._buf, len(self._buf) * 2)

    def _w(self, val: int) -> None:
        self._buf[self._cur] = val
        self._cur += 1

    def emit(
        self,
        kind: str,
        operands: tuple[IRValue, ...] = (),
        result_types: tuple[str, ...] = (),
        attrs: dict[str, Any] | None = None,
    ) -> tuple[IRValue, ...]:
        a = attrs or {}
        results = tuple(self._mk(rt) for rt in result_types)

        # --- Qubit alloc (not encoded, just tracked) ---
        if kind in ("pulse.qudit_arg", "pulse.qudit_alloc"):
            idx = a.get("index", self._next_qubit)
            for r in results:
                self._qref_to_qubit[r.vid] = idx
            self._next_qubit = max(self._next_qubit, idx + 1)
            return results

        # --- get_drive_line → ALLOC_DRIVE ---
        if kind == "pulse.get_drive_line":
            qref_vid = operands[0].vid if operands else None
            qubit = self._qref_to_qubit.get(qref_vid, 0) if qref_vid else 0
            self._ensure(4)
            self._w(_header(_ALLOC_DRIVE, 3))
            self._w(qubit)
            self._w(results[0].vid)  # line_vid
            self._w(results[1].vid)  # tone_vid
            return results

        # --- get_readout_line → ALLOC_READOUT ---
        if kind == "pulse.get_readout_line":
            qref_vid = operands[0].vid if operands else None
            qubit = self._qref_to_qubit.get(qref_vid, 0) if qref_vid else 0
            self._ensure(4)
            self._w(_header(_ALLOC_READOUT, 3))
            self._w(qubit)
            self._w(results[0].vid)
            self._w(results[1].vid)
            return results

        # --- Waveform constructors ---
        wf_name = kind.removeprefix("pulse.")
        wf_code = _WF_MAP.get(wf_name)
        if wf_code is not None or wf_name in ("custom", "custom_samples"):
            rv = results[0].vid
            dur_raw = a.get("duration", 0)
            dur = dur_raw if isinstance(dur_raw, Parameter) else int(dur_raw)
            wf_attrs = {"waveform_type": wf_name}
            if not isinstance(dur, Parameter):
                wf_attrs["duration_vtu"] = dur
            if "amplitude" in a:
                wf_attrs["amplitude"] = a["amplitude"]
            for k, v in a.items():
                if k not in ("duration", "amplitude"):
                    wf_attrs[k] = v
            self._wf_attrs[rv] = wf_attrs

            if wf_code == _WF_GAUSSIAN:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                vals = [dur, amp_raw, sig_raw]
                self._emit_wf(_WF_GAUSSIAN, rv, vals)
            elif wf_code == _WF_SQUARE:
                amp_raw = a.get("amplitude", 0.0)
                if isinstance(amp_raw, Parameter):
                    vals = [dur, amp_raw, amp_raw]
                else:
                    re, im = _complex_parts(amp_raw)
                    vals: list = [dur, re, im]
                self._emit_wf(_WF_SQUARE, rv, vals)
            elif wf_code == _WF_DRAG:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                beta_raw = a.get("beta", 0.0)
                vals = [dur, amp_raw, sig_raw, beta_raw]
                self._emit_wf(_WF_DRAG, rv, vals)
            elif wf_code == _WF_COSINE:
                amp_raw = a.get("amplitude", 0.0)
                vals = [dur, amp_raw]
                self._emit_wf(_WF_COSINE, rv, vals)
            elif wf_code == _WF_TANH_RAMP:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                vals = [dur, amp_raw, sig_raw]
                self._emit_wf(_WF_TANH_RAMP, rv, vals)
            elif wf_code == _WF_GAUSS_SQUARE:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                rf_raw = a.get("width", a.get("risefall", 0))
                vals = [dur, amp_raw, sig_raw, rf_raw]
                self._emit_wf(_WF_GAUSS_SQUARE, rv, vals)
            else:
                vals = [dur]
                self._emit_wf(_WF_CUSTOM, rv, vals)
            return results

        # --- drive ---
        if kind == "pulse.drive":
            lv = operands[0].vid
            wv = operands[1].vid
            tv = operands[2].vid
            rlv = results[0].vid
            rtv = results[1].vid
            self._ensure(8)
            self._w(_header(_DRIVE, 7))
            self._w(lv)
            self._w(wv)
            self._w(tv)
            self._w(rlv)
            self._w(rtv)
            self._w(_UNSCHEDULED)  # start_vtu (set by C++ scheduler)
            self._w(_UNSCHEDULED)  # duration_vtu (set by C++ scheduler)
            return results

        # --- readout ---
        if kind == "pulse.readout":
            lv = operands[0].vid
            wv = operands[1].vid
            tv = operands[2].vid
            rlv = results[0].vid
            rtv = results[1].vid
            mv = results[2].vid
            self._ensure(7)
            self._w(_header(_READOUT, 6))
            self._w(lv)
            self._w(wv)
            self._w(tv)
            self._w(rlv)
            self._w(rtv)
            self._w(mv)
            return results

        # --- sync ---
        if kind == "pulse.sync":
            n = len(operands)
            payload_len = 1 + 3 * n
            self._ensure(1 + payload_len)
            self._w(_header(_SYNC, payload_len))
            self._w(n)
            _LINE_TYPES = {"drive_line": 0, "readout_line": 1}
            for j in range(n):
                in_vid = operands[j].vid
                out_vid = results[j].vid if j < len(results) else in_vid
                vtype = _LINE_TYPES.get(
                    results[j].vtype if j < len(results) else "drive_line", 0)
                self._w(in_vid)
                self._w(out_vid)
                self._w(vtype)
            return results

        # --- wait ---
        if kind == "pulse.wait":
            lv = operands[0].vid
            rlv = results[0].vid
            dur_raw = a.get("duration", 0)
            if isinstance(dur_raw, Parameter):
                pvid = self._get_param_vid(dur_raw)
                self._ensure(4)
                self._w(_header(_WAIT, 3, 1 << 2))
                self._w(lv)
                self._w(rlv)
                self._w(pvid)
            else:
                dur = int(dur_raw)
                self._ensure(4)
                self._w(_header(_WAIT, 3))
                self._w(lv)
                self._w(rlv)
                self._w(dur)
            return results

        # --- shift_phase ---
        if kind == "pulse.shift_phase":
            tv = operands[0].vid
            rtv = results[0].vid
            val_raw = a.get("phase_rad", 0.0)
            if isinstance(val_raw, Parameter):
                pvid = self._get_param_vid(val_raw)
                self._ensure(4)
                self._w(_header(_SHIFT_PHASE, 3, 1 << 2))
                self._w(tv)
                self._w(rtv)
                self._w(pvid)
            else:
                delta = float(val_raw)
                self._ensure(4)
                self._w(_header(_SHIFT_PHASE, 3))
                self._w(tv)
                self._w(rtv)
                self._w(_f2i(delta))
            return results

        # --- set_phase ---
        if kind == "pulse.set_phase":
            tv = operands[0].vid
            rtv = results[0].vid
            val_raw = a.get("phase_rad", 0.0)
            if isinstance(val_raw, Parameter):
                pvid = self._get_param_vid(val_raw)
                self._ensure(4)
                self._w(_header(_SET_PHASE, 3, 1 << 2))
                self._w(tv)
                self._w(rtv)
                self._w(pvid)
            else:
                phase = float(val_raw)
                self._ensure(4)
                self._w(_header(_SET_PHASE, 3))
                self._w(tv)
                self._w(rtv)
                self._w(_f2i(phase))
            return results

        # --- shift_frequency ---
        if kind == "pulse.shift_frequency":
            tv = operands[0].vid
            rtv = results[0].vid
            val_raw = a.get("freq_hz", 0.0)
            if isinstance(val_raw, Parameter):
                pvid = self._get_param_vid(val_raw)
                self._ensure(4)
                self._w(_header(_SHIFT_FREQ, 3, 1 << 2))
                self._w(tv)
                self._w(rtv)
                self._w(pvid)
            else:
                freq = float(val_raw)
                self._ensure(4)
                self._w(_header(_SHIFT_FREQ, 3))
                self._w(tv)
                self._w(rtv)
                self._w(_f2i(freq))
            return results

        # --- set_frequency ---
        if kind == "pulse.set_frequency":
            tv = operands[0].vid
            rtv = results[0].vid
            val_raw = a.get("freq_hz", 0.0)
            if isinstance(val_raw, Parameter):
                pvid = self._get_param_vid(val_raw)
                self._ensure(4)
                self._w(_header(_SET_FREQ, 3, 1 << 2))
                self._w(tv)
                self._w(rtv)
                self._w(pvid)
            else:
                freq = float(val_raw)
                self._ensure(4)
                self._w(_header(_SET_FREQ, 3))
                self._w(tv)
                self._w(rtv)
                self._w(_f2i(freq))
            return results

        # --- SCF ops (loops/if) are structural, not encoded ---
        if kind.startswith("scf."):
            return results

        # Wf algebra ops: encode as custom for now
        if kind.startswith("pulse.wf_"):
            rv = results[0].vid if results else 0
            self._ensure(3)
            self._w(_header(_WF_CUSTOM, 2))
            self._w(rv)
            self._w(0)
            return results

        return results

    def _get_param_vid(self, param: Parameter) -> int:
        """Get or create a PARAM record for a Parameter, return its vid."""
        if param.index in self._param_vids:
            return self._param_vids[param.index]
        vid = self._next_id
        self._next_id += 1
        self._param_vids[param.index] = vid
        self._ensure(3)
        self._w(_header(_PARAM, 2))
        self._w(vid)
        self._w(param.index)
        return vid

    def _emit_wf(self, wf_code: int, rv: int, vals: list) -> None:
        """Emit a waveform record, handling mixed Parameter/concrete values.

        For each value slot, if the value is a Parameter, emit a PARAM record
        first and encode the param vid. A param_mask bit flags parametric slots
        so the C++ decoder can distinguish vids from literals.
        """
        encoded: list[int] = []
        param_mask = 0
        for i, v in enumerate(vals):
            if isinstance(v, Parameter):
                pvid = self._get_param_vid(v)
                encoded.append(pvid)
                param_mask |= (1 << (i + 1))  # +1 because slot 0 is the rv
            elif isinstance(v, int) and not isinstance(v, bool):
                encoded.append(v)
            else:
                encoded.append(_f2i(float(v)))
        payload_len = 1 + len(encoded)  # rv + values
        self._ensure(1 + payload_len)
        self._w(_header(wf_code, payload_len, param_mask))
        self._w(rv)
        for e in encoded:
            self._w(e)

    @property
    def param_names(self) -> list[str]:
        """Return parameter names in index order."""
        if not self._param_vids:
            return []
        max_idx = max(self._param_vids.keys())
        names: list[str] = [""] * (max_idx + 1)
        return names

    @property
    def has_parameters(self) -> bool:
        return bool(self._param_vids)

    def get_buffer(self) -> np.ndarray:
        """Return the trimmed packed buffer."""
        return self._buf[:self._cur].copy()

    def get_freq_array(self) -> np.ndarray:
        """Return qubit frequencies as a float64 array indexed by qubit."""
        arr = np.zeros(self._next_qubit, dtype=np.float64)
        for q, f in self._freq_hz.items():
            if q < self._next_qubit:
                arr[q] = f
        return arr

    @property
    def n_qubits(self) -> int:
        return self._next_qubit

    def pretty(self) -> str:
        return f"<PackedIRBuilder: {self._cur} words, {self._next_qubit} qubits>"
