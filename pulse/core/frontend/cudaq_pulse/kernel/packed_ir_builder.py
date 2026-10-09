# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
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

from .ir_builder import CompilationError, IRValue, Parameter, ParameterExpression, is_symbolic

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
_NUM_CONST = 19
_NUM_BINARY = 20
_NUM_NEG = 21
_NUM_CAST = 22
_WF_CUSTOM_SAMPLES = 23
_WF_ADD = 24
_WF_SUB = 25
_WF_MUL = 26
_WF_SCALE = 27
_WF_NEG = 28

_NUMERIC_TYPE = {"i64": 0, "f64": 1}
_BINARY_OP = {
    "add": 0,
    "sub": 1,
    "mul": 2,
    "div": 3,
    "floordiv": 4,
    "mod": 5,
}

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


def _as_i64(value: Any, description: str) -> int:
    """Convert an integer-valued Python number without silent truncation."""
    if isinstance(value, bool):
        raise CompilationError(f"{description} must be an integer, got bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    raise CompilationError(
        f"{description} must be an integer number of virtual time units, got {value!r}"
    )


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
        self._op_count = 0
        self._qref_to_qubit: dict[int, int] = {}
        self._wf_attrs: dict[int, dict[str, Any]] = {}
        self._param_vids: dict[int, int] = {}  # Parameter.index → vid
        # Key by the expression object itself, not id(expression): temporary
        # expressions may be collected and Python may reuse their integer IDs.
        self._numeric_vids: dict[tuple[ParameterExpression, str], int] = {}

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
        self._op_count += 1

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
            qubit = self._qref_to_qubit.get(qref_vid,
                                            0) if qref_vid is not None else 0
            self._ensure(4)
            self._w(_header(_ALLOC_DRIVE, 3))
            self._w(qubit)
            self._w(results[0].vid)  # line_vid
            self._w(results[1].vid)  # tone_vid
            return results

        # --- get_readout_line → ALLOC_READOUT ---
        if kind == "pulse.get_readout_line":
            qref_vid = operands[0].vid if operands else None
            qubit = self._qref_to_qubit.get(qref_vid,
                                            0) if qref_vid is not None else 0
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
            dur = dur_raw if is_symbolic(dur_raw) else _as_i64(
                dur_raw, f"{wf_name} duration")
            wf_attrs = {"waveform_type": wf_name}
            if not is_symbolic(dur):
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
                self._emit_wf(_WF_GAUSSIAN, rv, vals, ["i64", "f64", "f64"])
            elif wf_code == _WF_SQUARE:
                amp_raw = a.get("amplitude", 0.0)
                if is_symbolic(amp_raw):
                    vals = [dur, amp_raw, 0.0]
                else:
                    re, im = _complex_parts(amp_raw)
                    vals: list = [dur, re, im]
                self._emit_wf(_WF_SQUARE, rv, vals, ["i64", "f64", "f64"])
            elif wf_code == _WF_DRAG:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                beta_raw = a.get("beta", 0.0)
                vals = [dur, amp_raw, sig_raw, beta_raw]
                self._emit_wf(_WF_DRAG, rv, vals, ["i64", "f64", "f64", "f64"])
            elif wf_code == _WF_COSINE:
                amp_raw = a.get("amplitude", 0.0)
                vals = [dur, amp_raw]
                self._emit_wf(_WF_COSINE, rv, vals, ["i64", "f64"])
            elif wf_code == _WF_TANH_RAMP:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                vals = [dur, amp_raw, sig_raw]
                self._emit_wf(_WF_TANH_RAMP, rv, vals, ["i64", "f64", "f64"])
            elif wf_code == _WF_GAUSS_SQUARE:
                amp_raw = a.get("amplitude", 0.0)
                sig_raw = a.get("sigma", 1.0)
                width_raw = a.get("width", 0)
                # The Python API specifies flat-top width while the dialect
                # stores the duration of each rise/fall edge.
                if not is_symbolic(width_raw):
                    width_raw = _as_i64(width_raw,
                                        "gaussian_square flat-top width")
                if not is_symbolic(dur) and not is_symbolic(width_raw):
                    if width_raw < 0 or width_raw >= dur:
                        raise CompilationError(
                            "gaussian_square width must satisfy "
                            f"0 <= width < duration, got {width_raw} and {dur}")
                    if (dur - width_raw) % 2:
                        raise CompilationError(
                            "gaussian_square requires duration - width to be "
                            "even so its two edges have equal integer length")
                rf_raw = (dur - width_raw) // 2
                vals = [dur, amp_raw, sig_raw, rf_raw]
                self._emit_wf(_WF_GAUSS_SQUARE, rv, vals,
                              ["i64", "f64", "f64", "i64"])
            elif wf_name == "custom_samples":
                self._emit_custom_samples(rv, a.get("samples", ()))
            else:
                self._emit_custom(rv, dur, a.get("name", "custom"))
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
            if is_symbolic(dur_raw):
                pvid = self._get_numeric_vid(dur_raw, "i64")
                self._ensure(4)
                self._w(_header(_WAIT, 3, 1 << 2))
                self._w(lv)
                self._w(rlv)
                self._w(pvid)
            else:
                dur = _as_i64(dur_raw, "wait duration")
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
            if is_symbolic(val_raw):
                pvid = self._get_numeric_vid(val_raw, "f64")
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
            if is_symbolic(val_raw):
                pvid = self._get_numeric_vid(val_raw, "f64")
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
            if is_symbolic(val_raw):
                pvid = self._get_numeric_vid(val_raw, "f64")
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
            if is_symbolic(val_raw):
                pvid = self._get_numeric_vid(val_raw, "f64")
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

        # SCF operations are rejected or unrolled by the emitter.
        if kind.startswith("scf."):
            raise CompilationError(
                f"structured control flow op {kind!r} cannot be packed")

        algebra_opcodes = {
            "pulse.wf_add": _WF_ADD,
            "pulse.wf_sub": _WF_SUB,
            "pulse.wf_mul": _WF_MUL,
            "pulse.wf_neg": _WF_NEG,
        }
        if kind in algebra_opcodes:
            rv = results[0].vid
            self._ensure(2 + len(operands))
            self._w(_header(algebra_opcodes[kind], 1 + len(operands)))
            self._w(rv)
            for operand in operands:
                self._w(operand.vid)
            return results

        if kind == "pulse.wf_scale":
            rv = results[0].vid
            scale_vid = self._get_numeric_vid(a["scale"], "f64")
            self._ensure(4)
            self._w(_header(_WF_SCALE, 3))
            self._w(rv)
            self._w(operands[0].vid)
            self._w(scale_vid)
            return results

        raise CompilationError(f"unsupported packed pulse op: {kind}")

    def _get_parameter_base_vid(self, param: Parameter,
                                expected_dtype: str) -> int:
        """Get a block-argument reference and infer its storage type."""
        if param.dtype == "unknown":
            param.dtype = expected_dtype
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

    def _get_numeric_vid(self, value: Any, expected_dtype: str) -> int:
        """Materialize a symbolic or literal number as an SSA value record."""
        if expected_dtype not in _NUMERIC_TYPE:
            raise CompilationError(f"unsupported numeric type {expected_dtype}")

        if isinstance(value, Parameter):
            base = self._get_parameter_base_vid(value, expected_dtype)
            if value.dtype == expected_dtype:
                return base
            return self._emit_numeric_cast(base, value.dtype, expected_dtype)

        if isinstance(value, ParameterExpression):
            cache_key = (value, expected_dtype)
            if cache_key in self._numeric_vids:
                return self._numeric_vids[cache_key]

            if value.op == "cast":
                target_dtype = value.dtype
                source_dtype = getattr(value.lhs, "dtype", "unknown")
                if source_dtype == "unknown":
                    source_dtype = "f64" if target_dtype == "i64" else "i64"
                source = self._get_numeric_vid(value.lhs, source_dtype)
                result = self._emit_numeric_cast(source, source_dtype,
                                                 target_dtype)
                if target_dtype != expected_dtype:
                    result = self._emit_numeric_cast(result, target_dtype,
                                                     expected_dtype)
                self._numeric_vids[cache_key] = result
                return result

            dtype = "f64" if value.op == "div" else expected_dtype
            if value.op in ("floordiv", "mod"):
                dtype = "i64"
            if value.op == "pow":
                raise CompilationError(
                    "symbolic exponentiation is not supported; specialize "
                    "that value before compilation")
            if value.op == "neg":
                operand = self._get_numeric_vid(value.lhs, dtype)
                result = self._next_id
                self._next_id += 1
                self._ensure(4)
                self._w(_header(_NUM_NEG, 3))
                self._w(result)
                self._w(_NUMERIC_TYPE[dtype])
                self._w(operand)
            else:
                opcode = _BINARY_OP.get(value.op)
                if opcode is None:
                    raise CompilationError(
                        f"unsupported symbolic operation {value.op!r}")
                lhs = self._get_numeric_vid(value.lhs, dtype)
                rhs = self._get_numeric_vid(value.rhs, dtype)
                result = self._next_id
                self._next_id += 1
                self._ensure(6)
                self._w(_header(_NUM_BINARY, 5))
                self._w(result)
                self._w(_NUMERIC_TYPE[dtype])
                self._w(opcode)
                self._w(lhs)
                self._w(rhs)
            if dtype != expected_dtype:
                result = self._emit_numeric_cast(result, dtype, expected_dtype)
            self._numeric_vids[cache_key] = result
            return result

        result = self._next_id
        self._next_id += 1
        self._ensure(4)
        self._w(_header(_NUM_CONST, 3))
        self._w(result)
        self._w(_NUMERIC_TYPE[expected_dtype])
        self._w(
            _as_i64(value, "integer expression") if expected_dtype ==
            "i64" else _f2i(float(value)))
        return result

    def _emit_numeric_cast(self, operand: int, source_dtype: str,
                           target_dtype: str) -> int:
        if source_dtype == target_dtype:
            return operand
        result = self._next_id
        self._next_id += 1
        self._ensure(5)
        self._w(_header(_NUM_CAST, 4))
        self._w(result)
        self._w(_NUMERIC_TYPE[source_dtype])
        self._w(_NUMERIC_TYPE[target_dtype])
        self._w(operand)
        return result

    def _emit_wf(self, wf_code: int, rv: int, vals: list,
                 dtypes: list[str]) -> None:
        """Emit a waveform record, handling mixed Parameter/concrete values.

        For each value slot, if the value is a Parameter, emit a PARAM record
        first and encode the param vid. A param_mask bit flags parametric slots
        so the C++ decoder can distinguish vids from literals.
        """
        encoded: list[int] = []
        param_mask = 0
        for i, (v, dtype) in enumerate(zip(vals, dtypes)):
            if is_symbolic(v):
                pvid = self._get_numeric_vid(v, dtype)
                encoded.append(pvid)
                param_mask |= 1 << (i + 1)  # +1 because slot 0 is the rv
            elif dtype == "i64":
                encoded.append(_as_i64(v, "integer operand"))
            else:
                encoded.append(_f2i(float(v)))
        payload_len = 1 + len(encoded)  # rv + values
        self._ensure(1 + payload_len)
        self._w(_header(wf_code, payload_len, param_mask))
        self._w(rv)
        for e in encoded:
            self._w(e)

    def _emit_custom(self, rv: int, duration: Any, callback: Any) -> None:
        name = callback if isinstance(callback, str) else getattr(
            callback, "__name__", "custom")
        encoded = name.encode("utf-8")
        if not encoded:
            raise CompilationError("custom waveform callback name is empty")
        padded = encoded + b"\0" * ((8 - len(encoded) % 8) % 8)
        words = [
            struct.unpack("=q", padded[i:i + 8])[0]
            for i in range(0, len(padded), 8)
        ]
        if 3 + len(words) > 255:
            raise CompilationError("custom waveform callback name is too long")
        param_mask = 0
        if is_symbolic(duration):
            duration_word = self._get_numeric_vid(duration, "i64")
            param_mask |= 1 << 1
        else:
            duration_word = _as_i64(duration, "custom waveform duration")
        self._ensure(4 + len(words))
        self._w(_header(_WF_CUSTOM, 3 + len(words), param_mask))
        self._w(rv)
        self._w(duration_word)
        self._w(len(encoded))
        for word in words:
            self._w(word)

    def _emit_custom_samples(self, rv: int, samples: Any) -> None:
        try:
            sample_values = list(samples)
        except TypeError as exc:
            raise CompilationError(
                "custom_samples() requires a finite sequence") from exc
        if len(sample_values) > 253:
            raise CompilationError(
                "custom_samples() supports at most 253 samples per waveform")
        if not sample_values:
            raise CompilationError(
                "custom_samples() requires at least one sample")
        encoded: list[int] = []
        for sample in sample_values:
            value = complex(sample)
            if value.imag != 0.0:
                raise CompilationError(
                    "complex custom samples are not supported yet; provide "
                    "real-valued envelope samples")
            encoded.append(_f2i(value.real))
        self._ensure(3 + len(encoded))
        self._w(_header(_WF_CUSTOM_SAMPLES, 2 + len(encoded)))
        self._w(rv)
        self._w(len(encoded))
        for word in encoded:
            self._w(word)

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

    @property
    def op_count(self) -> int:
        return self._op_count

    def pretty(self) -> str:
        return f"<PackedIRBuilder: {self._cur} words, {self._next_qubit} qubits>"
