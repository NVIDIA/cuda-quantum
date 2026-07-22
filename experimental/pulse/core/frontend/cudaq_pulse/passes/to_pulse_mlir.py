# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit Pulse dialect MLIR text from an optimized Python Program.

This is the bridge between the Python pass IR and the C++ MLIR lowering
stack.  It takes a `Program` (after verify, canonicalize, virtual-z,
fusion, LICM, scheduling) and emits syntactically-valid pulse dialect
MLIR text that can be parsed by ``mlir-opt`` with the pulse dialect
registered.

The emitted text is a ``builtin.module { func.func @<name>(...) { ... } }``
wrapper around the pulse ops.
"""
from __future__ import annotations

import math
from typing import Any

from .ir_types import Op, OpKind, Program, Value, ValueType

_WAVEFORM_TYPE = "!pulse.waveform"
_DRIVE_LINE = "!pulse.drive_line"
_READOUT_LINE = "!pulse.readout_line"
_TONE = "!pulse.tone"
_QREF = "!pulse.qref"
_MEASUREMENT = "!pulse.measurement"
_DURATION = "!pulse.duration"

_VTYPE_TO_MLIR = {
    ValueType.DRIVE_LINE: _DRIVE_LINE,
    ValueType.READOUT_LINE: _READOUT_LINE,
    ValueType.TONE: _TONE,
    ValueType.WAVEFORM: _WAVEFORM_TYPE,
    ValueType.IQ_DATA: "!pulse.iq_data",
    ValueType.MEASUREMENT: _MEASUREMENT,
    ValueType.QREF: _QREF,
}


class _EmitterState:
    """Tracks SSA names, indentation, and qubit allocation during emission."""

    __slots__ = ("lines", "vid_to_ssa", "qubit_ssa", "indent", "_ssa_counter")

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.vid_to_ssa: dict[int, str] = {}
        self.qubit_ssa: dict[int, str] = {}
        self.indent: int = 2
        self._ssa_counter: int = 0

    def fresh_ssa(self, hint: str = "") -> str:
        name = f"%{hint}{self._ssa_counter}" if hint else f"%{self._ssa_counter}"
        self._ssa_counter += 1
        return name

    def bind(self, vid: int, ssa: str) -> None:
        self.vid_to_ssa[vid] = ssa

    def ref(self, vid: int) -> str:
        if vid not in self.vid_to_ssa:
            fallback = self.fresh_ssa("unresolved")
            self.vid_to_ssa[vid] = fallback
        return self.vid_to_ssa[vid]

    def emit(self, text: str) -> None:
        self.lines.append(" " * self.indent + text)


def _fmt_f64(val: float) -> str:
    """Format a float as MLIR f64 literal."""
    if val == float("inf"):
        return "0x7FF0000000000000"
    if val == float("-inf"):
        return "0xFFF0000000000000"
    if math.isnan(val):
        return "0x7FF8000000000000"
    if val == 0.0 and math.copysign(1.0, val) < 0:
        return "-0.0"
    s = f"{val:.15e}"
    return s.replace("e+0", "e+0").replace("e-0", "e-0")


def _real_of(val: Any) -> float:
    """Extract the real part from a possibly-complex value."""
    if isinstance(val, complex):
        return val.real
    return float(val)


def _mlir_type(vtype: ValueType) -> str:
    return _VTYPE_TO_MLIR[vtype]


def _emit_qudit_allocs(prog: Program, st: _EmitterState) -> None:
    """Emit pulse.qudit_alloc for each qubit referenced by alloc_drive/readout ops."""
    seen_qubits: set[int] = set()
    for op in prog.ops:
        if op.kind in (OpKind.ALLOC_DRIVE, OpKind.ALLOC_READOUT):
            qi = op.attrs.get("qubit", -1)
            if qi not in seen_qubits:
                seen_qubits.add(qi)
                ssa = st.fresh_ssa("q")
                st.qubit_ssa[qi] = ssa
                st.emit(f"{ssa} = pulse.qudit_alloc : {_QREF}")


def _emit_waveform(op: Op, st: _EmitterState) -> None:
    """Emit a pulse.gaussian / pulse.square / etc. waveform construction op."""
    wf_type = op.attrs.get("waveform_type", "square")
    duration = int(op.attrs.get("duration_vtu", 0))
    amplitude = op.attrs.get("amplitude", 0.0)
    result_ssa = st.fresh_ssa("wf")
    st.bind(op.results[0].vid, result_ssa)

    if wf_type == "gaussian":
        sigma = float(op.attrs.get("sigma", 1.0))
        st.emit(f"{result_ssa} = pulse.gaussian {duration}, "
                f"{_fmt_f64(_real_of(amplitude))}, {_fmt_f64(sigma)} "
                f": {_WAVEFORM_TYPE}")
    elif wf_type == "square":
        if isinstance(amplitude, (list, tuple)):
            amp_str = "[" + ", ".join(
                _fmt_f64(float(a)) for a in amplitude) + "]"
        elif isinstance(amplitude, complex):
            amp_str = f"[{_fmt_f64(amplitude.real)}, {_fmt_f64(amplitude.imag)}]"
        else:
            amp_str = f"[{_fmt_f64(float(amplitude))}, {_fmt_f64(0.0)}]"
        st.emit(f"{result_ssa} = pulse.square {duration}, {amp_str} "
                f": {_WAVEFORM_TYPE}")
    elif wf_type == "drag":
        sigma = float(op.attrs.get("sigma", 1.0))
        beta = float(op.attrs.get("beta", 0.0))
        st.emit(f"{result_ssa} = pulse.drag {duration}, "
                f"{_fmt_f64(_real_of(amplitude))}, {_fmt_f64(sigma)}, "
                f"{_fmt_f64(beta)} : {_WAVEFORM_TYPE}")
    elif wf_type == "cosine":
        st.emit(f"{result_ssa} = pulse.cosine {duration}, "
                f"{_fmt_f64(_real_of(amplitude))} : {_WAVEFORM_TYPE}")
    elif wf_type == "tanh_ramp":
        sigma = float(op.attrs.get("sigma", 1.0))
        st.emit(f"{result_ssa} = pulse.tanh_ramp {duration}, "
                f"{_fmt_f64(_real_of(amplitude))}, {_fmt_f64(sigma)} "
                f": {_WAVEFORM_TYPE}")
    elif wf_type == "gaussian_square":
        sigma = float(op.attrs.get("sigma", 1.0))
        risefall = int(op.attrs.get("risefall", 0))
        st.emit(f"{result_ssa} = pulse.gaussian_square {duration}, "
                f"{_fmt_f64(_real_of(amplitude))}, {_fmt_f64(sigma)}, "
                f"{risefall} : {_WAVEFORM_TYPE}")
    else:
        st.emit(f"{result_ssa} = pulse.custom @{wf_type}, {duration} "
                f": {_WAVEFORM_TYPE}")


def _emit_alloc_drive(op: Op, st: _EmitterState) -> None:
    qi = op.attrs.get("qubit", 0)
    q_ssa = st.qubit_ssa[qi]
    line_ssa = st.fresh_ssa("d")
    tone_ssa = st.fresh_ssa("t")
    st.bind(op.results[0].vid, line_ssa)
    st.bind(op.results[1].vid, tone_ssa)
    st.emit(f"{line_ssa}, {tone_ssa} = pulse.get_drive_line {q_ssa} "
            f": ({_QREF}) -> ({_DRIVE_LINE}, {_TONE})")


def _emit_alloc_readout(op: Op, st: _EmitterState) -> None:
    qi = op.attrs.get("qubit", 0)
    q_ssa = st.qubit_ssa[qi]
    line_ssa = st.fresh_ssa("r")
    tone_ssa = st.fresh_ssa("rt")
    st.bind(op.results[0].vid, line_ssa)
    st.bind(op.results[1].vid, tone_ssa)
    st.emit(f"{line_ssa}, {tone_ssa} = pulse.get_readout_line {q_ssa} "
            f": ({_QREF}) -> ({_READOUT_LINE}, {_TONE})")


def _emit_drive(op: Op, st: _EmitterState) -> None:
    line_in = st.ref(op.operands[0].vid)
    wf_in = st.ref(op.operands[1].vid)
    tone_in = st.ref(op.operands[2].vid)
    line_out = st.fresh_ssa("d")
    tone_out = st.fresh_ssa("t")
    st.bind(op.results[0].vid, line_out)
    st.bind(op.results[1].vid, tone_out)

    attrs = ""
    sched_attrs = []
    for key in ("start_vtu", "duration_vtu"):
        if key in op.attrs:
            sched_attrs.append(f"{key} = {int(op.attrs[key])} : i64")
    if sched_attrs:
        attrs = " {" + ", ".join(sched_attrs) + "}"

    st.emit(
        f"{line_out}, {tone_out} = pulse.drive {line_in}, {wf_in}, {tone_in}"
        f"{attrs} : {_DRIVE_LINE}, {_WAVEFORM_TYPE}, {_TONE} "
        f"-> {_DRIVE_LINE}, {_TONE}")


def _emit_readout(op: Op, st: _EmitterState) -> None:
    line_in = st.ref(op.operands[0].vid)
    wf_in = st.ref(op.operands[1].vid)
    tone_in = st.ref(op.operands[2].vid)
    line_out = st.fresh_ssa("r")
    tone_out = st.fresh_ssa("rt")
    meas_out = st.fresh_ssa("m")
    st.bind(op.results[0].vid, line_out)
    st.bind(op.results[1].vid, tone_out)
    st.bind(op.results[2].vid, meas_out)
    mode = op.attrs.get("mode", "iq")
    st.emit(f"{line_out}, {tone_out}, {meas_out} = pulse.readout "
            f"{line_in}, {wf_in}, {tone_in}, \"{mode}\" "
            f": {_READOUT_LINE}, {_WAVEFORM_TYPE}, {_TONE} "
            f"-> {_READOUT_LINE}, {_TONE}, {_MEASUREMENT}")


def _emit_wait(op: Op, st: _EmitterState) -> None:
    line_in = st.ref(op.operands[0].vid)
    line_out = st.fresh_ssa("d")
    st.bind(op.results[0].vid, line_out)
    dur_vtu = int(op.attrs.get("duration_vtu", 0))
    dur_const = st.fresh_ssa("c")
    dur_ssa = st.fresh_ssa("dur")
    line_type = _mlir_type(op.operands[0].vtype)
    st.emit(f"{dur_const} = arith.constant {dur_vtu} : i64")
    st.emit(f"{dur_ssa} = pulse.duration_from_int {dur_const} "
            f": (i64) -> {_DURATION}")
    st.emit(f"{line_out} = pulse.wait {line_in}, {dur_ssa} "
            f": ({line_type}, {_DURATION}) -> {line_type}")


def _emit_sync(op: Op, st: _EmitterState) -> None:
    in_ssas = [st.ref(o.vid) for o in op.operands]
    in_types = [_mlir_type(o.vtype) for o in op.operands]
    out_ssas = []
    for r in op.results:
        s = st.fresh_ssa("s")
        st.bind(r.vid, s)
        out_ssas.append(s)
    out_types = [_mlir_type(r.vtype) for r in op.results]
    st.emit(f"{', '.join(out_ssas)} = pulse.sync {', '.join(in_ssas)} "
            f": {', '.join(in_types)} -> {', '.join(out_types)}")


def _emit_shift_phase(op: Op, st: _EmitterState) -> None:
    tone_in = st.ref(op.operands[0].vid)
    tone_out = st.fresh_ssa("t")
    if op.results:
        st.bind(op.results[0].vid, tone_out)
    delta = float(op.attrs.get("delta_rad", op.attrs.get("delta", 0.0)))
    delta_ssa = st.fresh_ssa("ph")
    st.emit(f"{delta_ssa} = arith.constant {_fmt_f64(delta)} : f64")
    st.emit(f"{tone_out} = pulse.shift_phase {tone_in}, {delta_ssa} "
            f": {_TONE}, f64 -> {_TONE}")


def _emit_set_phase(op: Op, st: _EmitterState) -> None:
    tone_in = st.ref(op.operands[0].vid)
    tone_out = st.fresh_ssa("t")
    if op.results:
        st.bind(op.results[0].vid, tone_out)
    phase = float(op.attrs.get("phase_rad", op.attrs.get("phase", 0.0)))
    phase_ssa = st.fresh_ssa("ph")
    st.emit(f"{phase_ssa} = arith.constant {_fmt_f64(phase)} : f64")
    st.emit(f"{tone_out} = pulse.set_phase {tone_in}, {phase_ssa} "
            f": {_TONE}, f64 -> {_TONE}")


def _find_end_for(ops: list[Op], start_idx: int) -> int:
    """Find the matching END_FOR for a FOR_LOOP at start_idx."""
    depth = 0
    for i in range(start_idx, len(ops)):
        if ops[i].kind == OpKind.FOR_LOOP:
            depth += 1
        elif ops[i].kind == OpKind.END_FOR:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"Unbalanced FOR_LOOP at op[{start_idx}]")


def _linear_types_only(values: tuple[Value, ...]) -> list[Value]:
    """Filter to only linear-typed values (drive_line, readout_line, tone)."""
    return [
        v for v in values if v.vtype in (ValueType.DRIVE_LINE,
                                         ValueType.READOUT_LINE, ValueType.TONE)
    ]


def _emit_for_loop(
    op: Op,
    ops: list[Op],
    idx: int,
    st: _EmitterState,
) -> int:
    """Emit scf.for region. Returns the index past the matching END_FOR."""
    lb = int(op.attrs.get("lb", 0))
    ub = int(op.attrs.get("ub", 1))
    step = int(op.attrs.get("step", 1))

    end_idx = _find_end_for(ops, idx)
    end_op = ops[end_idx]

    linear_results = _linear_types_only(end_op.results)

    init_vids: list[int] = []
    init_ssas: list[str] = []
    iter_types: list[str] = []
    iter_arg_ssas: list[str] = []
    pre_vid_for_result: list[int] = []

    for lr in linear_results:
        mlir_t = _mlir_type(lr.vtype)
        found_pre = False
        for pre_op in reversed(ops[:idx]):
            for res in pre_op.results:
                if res.vtype == lr.vtype and res.vid in st.vid_to_ssa:
                    init_ssas.append(st.ref(res.vid))
                    init_vids.append(res.vid)
                    iter_types.append(mlir_t)
                    arg_ssa = st.fresh_ssa("arg")
                    iter_arg_ssas.append(arg_ssa)
                    pre_vid_for_result.append(res.vid)
                    found_pre = True
                    break
            if found_pre:
                break

    lb_ssa = st.fresh_ssa("lb")
    ub_ssa = st.fresh_ssa("ub")
    step_ssa = st.fresh_ssa("step")
    st.emit(f"{lb_ssa} = arith.constant {lb} : index")
    st.emit(f"{ub_ssa} = arith.constant {ub} : index")
    st.emit(f"{step_ssa} = arith.constant {step} : index")

    iv_ssa = st.fresh_ssa("iv")

    if iter_arg_ssas:
        result_ssas = []
        for i, lr in enumerate(linear_results):
            s = st.fresh_ssa("loop")
            st.bind(lr.vid, s)
            result_ssas.append(s)
        result_str = ", ".join(result_ssas)
        init_str = ", ".join(init_ssas)
        iter_type_str = ", ".join(iter_types)
        iter_args_str = ", ".join(
            f"{a} : {t}" for a, t in zip(iter_arg_ssas, iter_types))
        st.emit(f"{result_str} = scf.for {iv_ssa} = {lb_ssa} to {ub_ssa} "
                f"step {step_ssa} iter_args({iter_args_str}) = ({init_str}) "
                f"-> ({iter_type_str}) {{")
    else:
        st.emit(f"scf.for {iv_ssa} = {lb_ssa} to {ub_ssa} step {step_ssa} {{")

    saved_bindings = dict(st.vid_to_ssa)
    for pre_vid, arg_ssa in zip(pre_vid_for_result, iter_arg_ssas):
        st.vid_to_ssa[pre_vid] = arg_ssa

    st.indent += 2
    body_idx = idx + 1
    while body_idx < end_idx:
        body_idx = _emit_op(ops, body_idx, st)

    if iter_arg_ssas:
        yield_vals = []
        for lr in linear_results:
            for body_i in range(end_idx - 1, idx, -1):
                body_op = ops[body_i]
                for res in body_op.results:
                    if res.vtype == lr.vtype and res.vid in st.vid_to_ssa:
                        yield_vals.append(st.ref(res.vid))
                        break
                else:
                    continue
                break
            else:
                yield_vals.append(iter_arg_ssas[len(yield_vals)])
        yield_types = ", ".join(iter_types)
        yield_str = ", ".join(yield_vals[:len(iter_arg_ssas)])
        st.emit(f"scf.yield {yield_str} : {yield_types}")

    st.indent -= 2
    st.emit("}")

    for k, v in saved_bindings.items():
        if k not in st.vid_to_ssa:
            st.vid_to_ssa[k] = v

    return end_idx + 1


def _emit_op(ops: list[Op], idx: int, st: _EmitterState) -> int:
    """Emit a single op. Returns the next index to process."""
    op = ops[idx]

    if op.kind == OpKind.ALLOC_DRIVE:
        _emit_alloc_drive(op, st)
    elif op.kind == OpKind.ALLOC_READOUT:
        _emit_alloc_readout(op, st)
    elif op.kind == OpKind.ALLOC_TONE:
        freq = float(op.attrs.get("frequency_hz", 0.0))
        phase = float(op.attrs.get("phase_rad", 0.0))
        freq_ssa = st.fresh_ssa("freq")
        phase_ssa = st.fresh_ssa("ph")
        tone_ssa = st.fresh_ssa("t")
        st.bind(op.results[0].vid, tone_ssa)
        st.emit(f"{freq_ssa} = arith.constant {_fmt_f64(freq)} : f64")
        st.emit(f"{phase_ssa} = arith.constant {_fmt_f64(phase)} : f64")
        st.emit(f"{tone_ssa} = pulse.tone {freq_ssa}, {phase_ssa} "
                f": f64, f64 -> {_TONE}")
    elif op.kind == OpKind.MAKE_WAVEFORM:
        _emit_waveform(op, st)
    elif op.kind == OpKind.DRIVE:
        _emit_drive(op, st)
    elif op.kind == OpKind.READOUT:
        _emit_readout(op, st)
    elif op.kind == OpKind.WAIT:
        _emit_wait(op, st)
    elif op.kind == OpKind.SYNC:
        _emit_sync(op, st)
    elif op.kind == OpKind.SHIFT_PHASE:
        _emit_shift_phase(op, st)
    elif op.kind == OpKind.SET_PHASE:
        _emit_set_phase(op, st)
    elif op.kind == OpKind.FOR_LOOP:
        return _emit_for_loop(op, ops, idx, st)
    elif op.kind == OpKind.END_FOR:
        pass
    else:
        st.emit(f"// unsupported: {op.kind}")
    return idx + 1


def program_to_pulse_mlir(prog: Program) -> str:
    """Convert an optimized Program to pulse dialect MLIR text.

    Parameters
    ----------
    prog : Program
        The optimized program (post-verify, canonicalize, virtual-z,
        fusion, LICM, scheduling).

    Returns
    -------
    str
        MLIR module text parseable by ``mlir-opt`` with the pulse dialect.
    """
    st = _EmitterState()

    st.lines.append(f"module @{prog.name} {{")
    st.lines.append("  func.func @main() {")

    _emit_qudit_allocs(prog, st)

    idx = 0
    while idx < len(prog.ops):
        idx = _emit_op(prog.ops, idx, st)

    st.indent = 2
    st.emit("return")
    st.lines.append("  }")
    st.lines.append("}")

    return "\n".join(st.lines) + "\n"
