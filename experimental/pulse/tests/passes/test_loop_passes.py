# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudaq_pulse.passes.loop_passes import run_licm, run_loop_strength_reduction
from cudaq_pulse.passes.ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    _reset_vid_counter,
)


def _build_loop_program():
    """Program with a for-loop containing a hoistable MAKE_WAVEFORM."""
    _reset_vid_counter(500)
    d = Value(vid=500, vtype=ValueType.DRIVE_LINE, name="d0")
    t = Value(vid=501, vtype=ValueType.TONE, name="t0")
    wf = Value(vid=502, vtype=ValueType.WAVEFORM, name="wf")
    d2 = Value(vid=503, vtype=ValueType.DRIVE_LINE, name="d0")
    t2 = Value(vid=504, vtype=ValueType.TONE, name="t0")

    return Program(
        name="loop_test",
        clock_ghz=2.0,
        ops=[
            Op(kind=OpKind.ALLOC_DRIVE,
               operands=(),
               results=(d, t),
               attrs={
                   "qubit": 0,
                   "frequency_hz": 5e9
               }),
            Op(kind=OpKind.FOR_LOOP,
               operands=(),
               results=(),
               attrs={
                   "lb": 0,
                   "ub": 5,
                   "step": 1
               }),
            Op(kind=OpKind.MAKE_WAVEFORM,
               operands=(),
               results=(wf,),
               attrs={
                   "waveform_type": "gaussian",
                   "duration_vtu": 40,
                   "amplitude": 0.3,
                   "sigma": 10.0
               }),
            Op(kind=OpKind.DRIVE,
               operands=(d, wf, t),
               results=(d2, t2),
               attrs={"duration_vtu": 40}),
            Op(kind=OpKind.END_FOR, operands=(), results=(), attrs={}),
        ],
        values=[d, t, wf, d2, t2],
        qubit_freq_hz={0: 5e9},
    )


def _build_shift_phase_loop():
    """Program with a linear shift_phase progression in a loop."""
    _reset_vid_counter(600)
    d = Value(vid=600, vtype=ValueType.DRIVE_LINE, name="d0")
    t = Value(vid=601, vtype=ValueType.TONE, name="t0")
    t2 = Value(vid=602, vtype=ValueType.TONE, name="t0")

    return Program(
        name="shift_loop",
        clock_ghz=2.0,
        ops=[
            Op(kind=OpKind.ALLOC_DRIVE,
               operands=(),
               results=(d, t),
               attrs={
                   "qubit": 0,
                   "frequency_hz": 5e9
               }),
            Op(kind=OpKind.FOR_LOOP,
               operands=(),
               results=(),
               attrs={
                   "lb": 0,
                   "ub": 10,
                   "step": 1
               }),
            Op(kind=OpKind.SHIFT_PHASE,
               operands=(t,),
               results=(t2,),
               attrs={"delta_rad": 0.1}),
            Op(kind=OpKind.END_FOR, operands=(), results=(), attrs={}),
        ],
        values=[d, t, t2],
        qubit_freq_hz={0: 5e9},
    )


def test_licm_hoists_waveform():
    prog = _build_loop_program()
    result = run_licm(prog)
    for_idx = next(
        i for i, op in enumerate(result.ops) if op.kind == OpKind.FOR_LOOP)
    make_wf_before = [
        i for i, op in enumerate(result.ops)
        if op.kind == OpKind.MAKE_WAVEFORM and i < for_idx
    ]
    assert len(
        make_wf_before) >= 1, "LICM should hoist MAKE_WAVEFORM before loop"


def test_licm_preserves_echo(echo_program):
    result = run_licm(echo_program)
    assert result.op_count() > 0


def test_loop_strength_reduction_runs():
    prog = _build_shift_phase_loop()
    result = run_loop_strength_reduction(prog)
    assert result.op_count() > 0


def test_loop_strength_reduction_on_echo(echo_program):
    result = run_loop_strength_reduction(echo_program)
    assert result.op_count() > 0


def test_unbalanced_loop_raises():
    _reset_vid_counter(700)
    d = Value(vid=700, vtype=ValueType.DRIVE_LINE, name="d0")
    t = Value(vid=701, vtype=ValueType.TONE, name="t0")

    prog = Program(
        name="unbalanced",
        clock_ghz=2.0,
        ops=[
            Op(kind=OpKind.ALLOC_DRIVE,
               operands=(),
               results=(d, t),
               attrs={
                   "qubit": 0,
                   "frequency_hz": 5e9
               }),
            Op(kind=OpKind.FOR_LOOP,
               operands=(),
               results=(),
               attrs={
                   "lb": 0,
                   "ub": 5,
                   "step": 1
               }),
        ],
        values=[d, t],
        qubit_freq_hz={0: 5e9},
    )
    with pytest.raises(ValueError, match="FOR_LOOP.*without matching END_FOR"):
        run_licm(prog)
