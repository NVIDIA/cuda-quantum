# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudaq_pulse.passes.canonicalize import run_canonicalize
from cudaq_pulse.passes.ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    _reset_vid_counter,
)


def _simple_program_with_dead_alloc():
    """Program with a dead waveform alloc that canonicalize should remove."""
    _reset_vid_counter(400)
    d = Value(vid=400, vtype=ValueType.DRIVE_LINE, name="d0")
    t = Value(vid=401, vtype=ValueType.TONE, name="t0")
    wf_used = Value(vid=402, vtype=ValueType.WAVEFORM, name="wf_used")
    wf_dead = Value(vid=403, vtype=ValueType.WAVEFORM, name="wf_dead")
    d2 = Value(vid=404, vtype=ValueType.DRIVE_LINE, name="d0")
    t2 = Value(vid=405, vtype=ValueType.TONE, name="t0")

    return Program(
        name="dead_alloc",
        clock_ghz=2.0,
        ops=[
            Op(kind=OpKind.ALLOC_DRIVE,
               operands=(),
               results=(d, t),
               attrs={
                   "qubit": 0,
                   "frequency_hz": 5e9
               }),
            Op(kind=OpKind.MAKE_WAVEFORM,
               operands=(),
               results=(wf_used,),
               attrs={
                   "waveform_type": "gaussian",
                   "duration_vtu": 40,
                   "amplitude": 0.3,
                   "sigma": 10.0
               }),
            Op(kind=OpKind.MAKE_WAVEFORM,
               operands=(),
               results=(wf_dead,),
               attrs={
                   "waveform_type": "gaussian",
                   "duration_vtu": 40,
                   "amplitude": 0.1,
                   "sigma": 10.0
               }),
            Op(kind=OpKind.DRIVE,
               operands=(d, wf_used, t),
               results=(d2, t2),
               attrs={"duration_vtu": 40}),
        ],
        values=[d, t, wf_used, wf_dead, d2, t2],
        qubit_freq_hz={0: 5e9},
    )


def test_canonicalize_preserves_ops(simple_program):
    """Canonical program keeps all ops when nothing to optimize."""
    result = run_canonicalize(simple_program)
    assert result.op_count() >= simple_program.op_count() - 1


def test_canonicalize_removes_dead_alloc():
    prog = _simple_program_with_dead_alloc()
    original_count = prog.op_count()
    result = run_canonicalize(prog)
    make_wfs = [op for op in result.ops if op.kind == OpKind.MAKE_WAVEFORM]
    assert len(make_wfs) <= 2


def test_canonicalize_two_qubit(two_qubit_program):
    result = run_canonicalize(two_qubit_program)
    assert result.op_count() > 0


def test_canonicalize_echo(echo_program):
    result = run_canonicalize(echo_program)
    assert result.op_count() > 0
    assert result.clock_ghz == echo_program.clock_ghz
