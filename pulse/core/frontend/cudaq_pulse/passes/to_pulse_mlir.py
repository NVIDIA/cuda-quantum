# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
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
            raise ValueError(f"value %{vid} is used before it is defined")
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


def _integer_vtu(value: Any, description: str) -> int:
    """Return an integer VTU value without silently truncating floats."""
    if isinstance(value, bool):
        raise ValueError(f"{description} must be an integer, got bool")
    try:
        converted = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{description} must be an integer virtual-time value, got {value!r}"
        ) from exc
    if not math.isfinite(numeric) or numeric != converted:
        raise ValueError(
            f"{description} must be an integer virtual-time value, got {value!r}"
        )
    return converted


def _mlir_type(vtype: ValueType) -> str:
    return _VTYPE_TO_MLIR[vtype]


def _emit_qudit_allocs(prog: Program, st: _EmitterState) -> None:
    """Emit pulse.qudit_alloc for each qubit referenced by alloc_drive/readout ops."""
    seen_qubits: set[int] = set()
    for op in prog.ops:
        if op.kind in (OpKind.ALLOC_DRIVE, OpKind.ALLOC_READOUT):
            qi = op.attrs.get("qubit", -1)
            if not isinstance(qi, int) or qi < 0:
                raise ValueError(f"invalid physical qubit index {qi!r}")
            if qi not in seen_qubits:
                seen_qubits.add(qi)
                ssa = st.fresh_ssa("q")
                st.qubit_ssa[qi] = ssa
                st.emit(
                    f"{ssa} = pulse.qudit_alloc {{qubit = {qi} : i64}} : {_QREF}"
                )


def _emit_waveform(op: Op, st: _EmitterState) -> None:
    """Emit a pulse.gaussian / pulse.square / etc. waveform construction op."""
    wf_type = op.attrs.get("waveform_type", "square")
    duration = _integer_vtu(op.attrs.get("duration_vtu", 0),
                            f"{wf_type} duration")
    amplitude = op.attrs.get("amplitude", 0.0)
    result_ssa = st.fresh_ssa("wf")
    st.bind(op.results[0].vid, result_ssa)

    def constant(value: float | int, typ: str, hint: str) -> str:
        ssa = st.fresh_ssa(hint)
        literal = str(value) if typ == "i64" else _fmt_f64(float(value))
        st.emit(f"{ssa} = arith.constant {literal} : {typ}")
        return ssa

    duration_ssa = constant(duration, "i64", "dur")

    if wf_type == "gaussian":
        sigma = float(op.attrs.get("sigma", 1.0))
        amplitude_ssa = constant(_real_of(amplitude), "f64", "amp")
        sigma_ssa = constant(sigma, "f64", "sigma")
        st.emit(f"{result_ssa} = pulse.gaussian {duration_ssa}, "
                f"{amplitude_ssa}, {sigma_ssa} : i64, f64, f64 -> "
                f"{_WAVEFORM_TYPE}")
    elif wf_type == "square":
        if isinstance(amplitude, (list, tuple)):
            real = float(amplitude[0]) if amplitude else 0.0
            imaginary = float(amplitude[1]) if len(amplitude) > 1 else 0.0
        elif isinstance(amplitude, complex):
            real, imaginary = amplitude.real, amplitude.imag
        else:
            real, imaginary = float(amplitude), 0.0
        real_ssa = constant(real, "f64", "amp")
        imaginary_ssa = constant(imaginary, "f64", "amp")
        st.emit(f"{result_ssa} = pulse.square {duration_ssa}, {real_ssa}, "
                f"{imaginary_ssa} : i64, f64, f64 -> {_WAVEFORM_TYPE}")
    elif wf_type == "drag":
        sigma = float(op.attrs.get("sigma", 1.0))
        beta = float(op.attrs.get("beta", 0.0))
        amplitude_ssa = constant(_real_of(amplitude), "f64", "amp")
        sigma_ssa = constant(sigma, "f64", "sigma")
        beta_ssa = constant(beta, "f64", "beta")
        st.emit(f"{result_ssa} = pulse.drag {duration_ssa}, {amplitude_ssa}, "
                f"{sigma_ssa}, {beta_ssa} : i64, f64, f64, f64 -> "
                f"{_WAVEFORM_TYPE}")
    elif wf_type == "cosine":
        amplitude_ssa = constant(_real_of(amplitude), "f64", "amp")
        st.emit(f"{result_ssa} = pulse.cosine {duration_ssa}, {amplitude_ssa} "
                f": i64, f64 -> {_WAVEFORM_TYPE}")
    elif wf_type == "tanh_ramp":
        sigma = float(op.attrs.get("sigma", 1.0))
        amplitude_ssa = constant(_real_of(amplitude), "f64", "amp")
        sigma_ssa = constant(sigma, "f64", "sigma")
        st.emit(f"{result_ssa} = pulse.tanh_ramp {duration_ssa}, "
                f"{amplitude_ssa}, {sigma_ssa} : i64, f64, f64 -> "
                f"{_WAVEFORM_TYPE}")
    elif wf_type == "gaussian_square":
        sigma = float(op.attrs.get("sigma", 1.0))
        if "risefall" in op.attrs:
            risefall = _integer_vtu(op.attrs["risefall"],
                                    "gaussian_square rise/fall")
        else:
            width = _integer_vtu(op.attrs.get("width", 0),
                                 "gaussian_square width")
            if width < 0 or width >= duration:
                raise ValueError(
                    "gaussian_square width must satisfy "
                    f"0 <= width < duration, got {width} and {duration}")
            if (duration - width) % 2:
                raise ValueError(
                    "gaussian_square requires duration - width to be even "
                    "so its two edges have equal integer length")
            risefall = (duration - width) // 2
        amplitude_ssa = constant(_real_of(amplitude), "f64", "amp")
        sigma_ssa = constant(sigma, "f64", "sigma")
        risefall_ssa = constant(risefall, "i64", "edge")
        st.emit(f"{result_ssa} = pulse.gaussian_square {duration_ssa}, "
                f"{amplitude_ssa}, {sigma_ssa}, {risefall_ssa} : "
                f"i64, f64, f64, i64 -> {_WAVEFORM_TYPE}")
    elif wf_type == "custom_samples":
        samples = op.attrs.get("samples", ())
        sample_text = ", ".join(_fmt_f64(float(v)) for v in samples)
        st.emit(
            f"{result_ssa} = pulse.custom_samples [{sample_text}] : {_WAVEFORM_TYPE}"
        )
    else:
        name = str(op.attrs.get("name", wf_type)).lstrip("@")
        st.emit(
            f"{result_ssa} = pulse.custom @{name}, {duration_ssa} : i64 -> {_WAVEFORM_TYPE}"
        )


def _emit_alloc_drive(op: Op, st: _EmitterState) -> None:
    qi = op.attrs.get("qubit", 0)
    q_ssa = st.qubit_ssa[qi]
    line_ssa = st.fresh_ssa("d")
    tone_ssa = st.fresh_ssa("t")
    st.bind(op.results[0].vid, line_ssa)
    st.bind(op.results[1].vid, tone_ssa)
    frequency = float(op.attrs.get("frequency_hz", 0.0))
    st.emit(
        f"{line_ssa}, {tone_ssa} = pulse.get_drive_line {q_ssa} "
        f"{{qubit = {qi} : i64, frequency_hz = {_fmt_f64(frequency)} : f64}} "
        f": ({_QREF}) -> ({_DRIVE_LINE}, {_TONE})")


def _emit_alloc_readout(op: Op, st: _EmitterState) -> None:
    qi = op.attrs.get("qubit", 0)
    q_ssa = st.qubit_ssa[qi]
    line_ssa = st.fresh_ssa("r")
    tone_ssa = st.fresh_ssa("rt")
    st.bind(op.results[0].vid, line_ssa)
    st.bind(op.results[1].vid, tone_ssa)
    frequency = float(op.attrs.get("frequency_hz", 0.0))
    st.emit(
        f"{line_ssa}, {tone_ssa} = pulse.get_readout_line {q_ssa} "
        f"{{qubit = {qi} : i64, frequency_hz = {_fmt_f64(frequency)} : f64}} "
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
            value = _integer_vtu(op.attrs[key], f"drive {key}")
            sched_attrs.append(f"{key} = {value} : i64")
    if "phase_offset" in op.attrs or "phase" in op.attrs:
        phase = float(op.attrs.get("phase_offset", op.attrs.get("phase", 0.0)))
        sched_attrs.append(f"phase_offset = {_fmt_f64(phase)} : f64")
    if "frame_phase_offset" in op.attrs:
        frame_phase = float(op.attrs["frame_phase_offset"])
        sched_attrs.append(
            f"frame_phase_offset = {_fmt_f64(frame_phase)} : f64")
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
            f'{line_in}, {wf_in}, {tone_in}, "{mode}" '
            f": {_READOUT_LINE}, {_WAVEFORM_TYPE}, {_TONE} "
            f"-> {_READOUT_LINE}, {_TONE}, {_MEASUREMENT}")


def _emit_wait(op: Op, st: _EmitterState) -> None:
    line_in = st.ref(op.operands[0].vid)
    line_out = st.fresh_ssa("d")
    st.bind(op.results[0].vid, line_out)
    dur_vtu = _integer_vtu(op.attrs.get("duration_vtu", 0), "wait duration")
    dur_const = st.fresh_ssa("c")
    dur_ssa = st.fresh_ssa("dur")
    line_type = _mlir_type(op.operands[0].vtype)
    st.emit(f"{dur_const} = arith.constant {dur_vtu} : i64")
    st.emit(
        f"{dur_ssa} = pulse.duration_from_int {dur_const} : (i64) -> {_DURATION}"
    )
    st.emit(
        f"{line_out} = pulse.wait {line_in}, {dur_ssa} : ({line_type}, {_DURATION}) -> {line_type}"
    )


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
    st.emit(
        f"{tone_out} = pulse.shift_phase {tone_in}, {delta_ssa} : {_TONE}, f64 -> {_TONE}"
    )


def _emit_set_phase(op: Op, st: _EmitterState) -> None:
    tone_in = st.ref(op.operands[0].vid)
    tone_out = st.fresh_ssa("t")
    if op.results:
        st.bind(op.results[0].vid, tone_out)
    phase = float(op.attrs.get("phase_rad", op.attrs.get("phase", 0.0)))
    phase_ssa = st.fresh_ssa("ph")
    st.emit(f"{phase_ssa} = arith.constant {_fmt_f64(phase)} : f64")
    st.emit(
        f"{tone_out} = pulse.set_phase {tone_in}, {phase_ssa} : {_TONE}, f64 -> {_TONE}"
    )


def _emit_frequency_op(op: Op, st: _EmitterState, *, shift: bool) -> None:
    tone_in = st.ref(op.operands[0].vid)
    tone_out = st.fresh_ssa("t")
    if op.results:
        st.bind(op.results[0].vid, tone_out)
    frequency = float(op.attrs.get("frequency_hz", 0.0))
    frequency_ssa = st.fresh_ssa("freq")
    st.emit(f"{frequency_ssa} = arith.constant {_fmt_f64(frequency)} : f64")
    name = "shift_frequency" if shift else "set_frequency"
    st.emit(
        f"{tone_out} = pulse.{name} {tone_in}, {frequency_ssa} : {_TONE}, f64 -> {_TONE}"
    )


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
    lb = _integer_vtu(op.attrs.get("lb", 0), "loop lower bound")
    ub = _integer_vtu(op.attrs.get("ub", 1), "loop upper bound")
    step = _integer_vtu(op.attrs.get("step", 1), "loop step")

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
        st.emit(
            f"{tone_ssa} = pulse.tone {freq_ssa}, {phase_ssa} : f64, f64 -> {_TONE}"
        )
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
    elif op.kind == OpKind.SHIFT_FREQUENCY:
        _emit_frequency_op(op, st, shift=True)
    elif op.kind == OpKind.SET_FREQUENCY:
        _emit_frequency_op(op, st, shift=False)
    elif op.kind == OpKind.FOR_LOOP:
        return _emit_for_loop(op, ops, idx, st)
    elif op.kind == OpKind.END_FOR:
        pass
    else:
        raise ValueError(f"cannot emit unsupported pulse op {op.kind!r}")
    return idx + 1


def program_to_pulse_mlir(
    prog: Program,
    *,
    target: Any | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    num_steps: int | None = None,
    integrator: str | None = None,
) -> str:
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
    active_qubits = sorted(prog.qubit_freq_hz)
    n_qubits = max(active_qubits, default=0) + 1
    attrs = [
        f"pulse.clock_ghz = {_fmt_f64(prog.clock_ghz)} : f64",
        f"qop.n_qubits = {n_qubits} : i64",
    ]
    if active_qubits:
        frequencies = [prog.qubit_freq_hz.get(i, 0.0) for i in range(n_qubits)]
        attrs.append("pulse.qubit_freq_hz = array<f64: " +
                     ", ".join(_fmt_f64(value) for value in frequencies) + ">")
    if t_start is not None:
        attrs.append(f"qop.t_start = {_fmt_f64(t_start)} : f64")
    if t_end is not None:
        attrs.append(f"qop.t_end = {_fmt_f64(t_end)} : f64")
    if num_steps is not None:
        attrs.append(f"qop.num_steps = {int(num_steps)} : i64")
    if integrator is not None:
        attrs.append(f'qop.integrator = "{integrator}"')

    if target is not None:
        missing = [q for q in active_qubits if q not in target.qubits]
        if missing:
            raise ValueError(
                f"target {target.name!r} does not define active qubits {missing}"
            )
        t1_ns = [0.0] * n_qubits
        t2_ns = [0.0] * n_qubits
        drive_scales = [1.0] * n_qubits
        for qi in active_qubits:
            qubit = target.qubits[qi]
            t1_ns[qi] = max(0.0, float(qubit.t1_us)) * 1.0e3
            t2_ns[qi] = max(0.0, float(qubit.t2_star_us)) * 1.0e3
            drive_scales[qi] = target.drive_amplitude_scale(qi)
        attrs.append("pulse.t1_times = [" +
                     ", ".join(f"{_fmt_f64(value)} : f64" for value in t1_ns) +
                     "]")
        attrs.append("pulse.t2_times = [" +
                     ", ".join(f"{_fmt_f64(value)} : f64" for value in t2_ns) +
                     "]")
        attrs.append("pulse.drive_scale_rad_per_ns = array<f64: " +
                     ", ".join(_fmt_f64(value) for value in drive_scales) + ">")
        couplings = [
            c for c in target.couplings
            if c.qubit_a in active_qubits and c.qubit_b in active_qubits
        ]
        if couplings:
            pairs = [
                index for c in couplings for index in (c.qubit_a, c.qubit_b)
            ]
            attrs.append("pulse.coupling_pairs = array<i64: " +
                         ", ".join(str(value) for value in pairs) + ">")
            attrs.append("pulse.coupling_strength_hz = array<f64: " + ", ".join(
                _fmt_f64(c.coupling_strength_hz) for c in couplings) + ">")

        crosstalk = [
            c for c in target.crosstalk
            if c.qubit_a in active_qubits and c.qubit_b in active_qubits
        ]
        if crosstalk:
            pairs = [
                index for c in crosstalk for index in (c.qubit_a, c.qubit_b)
            ]
            attrs.append("pulse.crosstalk_pairs = array<i64: " +
                         ", ".join(str(value) for value in pairs) + ">")
            attrs.append("pulse.crosstalk_strength_hz = array<f64: " +
                         ", ".join(
                             _fmt_f64(c.static_zz_hz) for c in crosstalk) + ">")

    st.lines.append(f"module @{prog.name} attributes {{")
    for index, attr in enumerate(attrs):
        comma = "," if index + 1 < len(attrs) else ""
        st.lines.append(f"  {attr}{comma}")
    st.lines.append("} {")
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
