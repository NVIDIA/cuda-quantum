# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert kernel IR (PythonIRBuilder) to pass-level IR (Program).

This module bridges the gap between the frontend ``@cudaq_pulse.kernel``
compilation output and the passes that operate on ``Program`` objects.
"""

from __future__ import annotations

from typing import Any

from .kernel.ir_builder import (
    CompilationError,
    IRValue,
    Op as KernelOp,
    PythonIRBuilder,
)
from .passes.ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
)

_VTYPE_MAP: dict[str, ValueType] = {
    "drive_line": ValueType.DRIVE_LINE,
    "readout_line": ValueType.READOUT_LINE,
    "tone": ValueType.TONE,
    "waveform": ValueType.WAVEFORM,
    "qref": ValueType.QREF,
    "measurement": ValueType.MEASUREMENT,
    "iq_data": ValueType.IQ_DATA,
}

_WAVEFORM_OPS = frozenset({
    "pulse.gaussian",
    "pulse.square",
    "pulse.drag",
    "pulse.cosine",
    "pulse.tanh_ramp",
    "pulse.gaussian_square",
    "pulse.custom",
    "pulse.custom_samples",
})

_KIND_MAP: dict[str, str] = {
    "pulse.drive": OpKind.DRIVE,
    "pulse.readout": OpKind.READOUT,
    "pulse.wait": OpKind.WAIT,
    "pulse.sync": OpKind.SYNC,
    "pulse.shift_phase": OpKind.SHIFT_PHASE,
    "pulse.set_phase": OpKind.SET_PHASE,
    "scf.for": OpKind.FOR_LOOP,
    "scf.for_end": OpKind.END_FOR,
}


def _to_program(
    ir: PythonIRBuilder,
    *,
    clock_ghz: float = 2.0,
    qubit_freq_hz: dict[int, float] | None = None,
) -> Program:
    """Lower kernel IR to a pass-level Program.

    Parameters
    ----------
    ir : PythonIRBuilder
        The IR produced by calling a ``@cudaq_pulse.kernel`` function.
    clock_ghz : float
        System clock frequency in GHz (VTU scaling).
    qubit_freq_hz : dict[int, float] | None
        Mapping from qubit index to qubit frequency in Hz.
        When provided, ``get_drive_line`` / ``get_readout_line`` ops
        are annotated with the target frequency.
    """
    freq_hz = qubit_freq_hz or {}
    val_map: dict[int, Value] = {}
    qref_to_qubit: dict[int, int] = {}
    wf_attrs: dict[int, dict[str, Any]] = {}
    next_qubit = 0
    ops: list[Op] = []
    values: list[Value] = []
    collected_freqs: dict[int, float] = {}

    def _map_val(iv: IRValue) -> Value:
        if iv.vid in val_map:
            return val_map[iv.vid]
        vtype = _VTYPE_MAP.get(iv.vtype)
        if vtype is None:
            raise CompilationError(
                f"unknown IR value type {iv.vtype!r} for %{iv.vid} "
                f"(name={iv.name!r}); add it to _VTYPE_MAP in lower.py")
        v = Value(vid=iv.vid, vtype=vtype, name=iv.name)
        val_map[iv.vid] = v
        values.append(v)
        return v

    for kop in ir.ops:
        kind = kop.kind

        # Qubit argument / alloc → track qubit index
        if kind in ("pulse.qudit_arg", "pulse.qudit_alloc"):
            for r in kop.results:
                v = _map_val(r)
                idx = kop.attrs.get("index", next_qubit)
                qref_to_qubit[r.vid] = idx
            next_qubit = max(next_qubit, idx + 1) if kop.results else next_qubit
            continue

        # get_drive_line / get_readout_line → ALLOC
        if kind == "pulse.get_drive_line":
            qref_vid = kop.operands[0].vid if kop.operands else None
            qubit_idx = qref_to_qubit.get(qref_vid, 0) if qref_vid else 0
            if qubit_idx not in freq_hz:
                raise CompilationError(
                    f"no frequency provided for qubit {qubit_idx}; "
                    f"pass qubit_freq_hz={{...}} to to_program()")
            fhz = freq_hz[qubit_idx]
            collected_freqs[qubit_idx] = fhz
            result_vals = tuple(_map_val(r) for r in kop.results)
            ops.append(
                Op(
                    kind=OpKind.ALLOC_DRIVE,
                    operands=(),
                    results=result_vals,
                    attrs={
                        "qubit": qubit_idx,
                        "frequency_hz": fhz
                    },
                ))
            continue

        if kind == "pulse.get_readout_line":
            qref_vid = kop.operands[0].vid if kop.operands else None
            qubit_idx = qref_to_qubit.get(qref_vid, 0) if qref_vid else 0
            if qubit_idx not in freq_hz:
                raise CompilationError(
                    f"no frequency provided for qubit {qubit_idx}; "
                    f"pass qubit_freq_hz={{...}} to to_program()")
            fhz = freq_hz[qubit_idx]
            collected_freqs[qubit_idx] = fhz
            result_vals = tuple(_map_val(r) for r in kop.results)
            ops.append(
                Op(
                    kind=OpKind.ALLOC_READOUT,
                    operands=(),
                    results=result_vals,
                    attrs={
                        "qubit": qubit_idx,
                        "frequency_hz": fhz
                    },
                ))
            continue

        # Waveform constructors → MAKE_WAVEFORM
        if kind in _WAVEFORM_OPS:
            wf_type = kind.removeprefix("pulse.")
            result_vals = tuple(_map_val(r) for r in kop.results)
            attrs = {"waveform_type": wf_type}
            if "duration" in kop.attrs:
                attrs["duration_vtu"] = kop.attrs["duration"]
            if "amplitude" in kop.attrs:
                attrs["amplitude"] = kop.attrs["amplitude"]
            for k, v in kop.attrs.items():
                if k not in ("duration", "amplitude"):
                    attrs[k] = v
            if result_vals:
                wf_attrs[result_vals[0].vid] = attrs
            ops.append(
                Op(
                    kind=OpKind.MAKE_WAVEFORM,
                    operands=(),
                    results=result_vals,
                    attrs=attrs,
                ))
            continue

        # Drive / readout — annotate with duration from waveform
        if kind in ("pulse.drive", "pulse.readout"):
            operand_vals = tuple(_map_val(o) for o in kop.operands)
            result_vals = tuple(_map_val(r) for r in kop.results)
            attrs = dict(kop.attrs)
            for ov in operand_vals:
                if ov.vtype == ValueType.WAVEFORM and ov.vid in wf_attrs:
                    wa = wf_attrs[ov.vid]
                    attrs.setdefault("duration_vtu", wa.get("duration_vtu", 0))
                    attrs.setdefault("amplitude", wa.get("amplitude", 0))
                    attrs.setdefault("waveform_type",
                                     wa.get("waveform_type", ""))
            pass_kind = _KIND_MAP.get(kind, kind)
            ops.append(
                Op(
                    kind=pass_kind,
                    operands=operand_vals,
                    results=result_vals,
                    attrs=attrs,
                ))
            continue

        # Wait → copy duration attr
        if kind == "pulse.wait":
            operand_vals = tuple(_map_val(o) for o in kop.operands)
            result_vals = tuple(_map_val(r) for r in kop.results)
            attrs = {}
            if "duration" in kop.attrs:
                attrs["duration_vtu"] = kop.attrs["duration"]
            else:
                attrs.update(kop.attrs)
            ops.append(
                Op(
                    kind=OpKind.WAIT,
                    operands=operand_vals,
                    results=result_vals,
                    attrs=attrs,
                ))
            continue

        # Shift/set phase — rename attr key
        if kind in ("pulse.shift_phase", "pulse.set_phase"):
            operand_vals = tuple(_map_val(o) for o in kop.operands)
            result_vals = tuple(_map_val(r) for r in kop.results)
            attrs = {}
            if "phase_rad" in kop.attrs:
                if kind == "pulse.shift_phase":
                    attrs["delta_rad"] = kop.attrs["phase_rad"]
                else:
                    attrs["phase_rad"] = kop.attrs["phase_rad"]
            else:
                attrs.update(kop.attrs)
            pass_kind = _KIND_MAP[kind]
            ops.append(
                Op(
                    kind=pass_kind,
                    operands=operand_vals,
                    results=result_vals,
                    attrs=attrs,
                ))
            continue

        if kind in ("scf.if", "scf.else", "scf.if_end"):
            raise CompilationError(
                f"conditional control flow ({kind}) is not yet supported "
                f"in the pass IR. Mid-circuit measurement branching requires "
                f"IF_BEGIN/IF_END lowering (not yet implemented).")

        if kind == "scf.yield":
            continue

        # Generic mapping for remaining ops
        operand_vals = tuple(_map_val(o) for o in kop.operands)
        result_vals = tuple(_map_val(r) for r in kop.results)
        pass_kind = _KIND_MAP.get(kind, kind)
        ops.append(
            Op(
                kind=pass_kind,
                operands=operand_vals,
                results=result_vals,
                attrs=dict(kop.attrs),
            ))

    return Program(
        name=ir.name,
        clock_ghz=clock_ghz,
        ops=ops,
        values=values,
        qubit_freq_hz=collected_freqs or dict(freq_hz),
    )
