# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudaq_pulse.passes.ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    _mk,
    _reset_vid_counter,
    clone_program,
    duration_of,
    line_id_of,
    tone_id_of,
    waveform_of,
    is_loop_or_barrier,
)


def test_value_creation():
    v = Value(vid=0, vtype=ValueType.DRIVE_LINE, name="d0")
    assert v.vid == 0
    assert v.vtype == ValueType.DRIVE_LINE
    assert v.name == "d0"


def test_op_creation():
    v_in = Value(vid=0, vtype=ValueType.DRIVE_LINE, name="d0")
    v_out = Value(vid=1, vtype=ValueType.DRIVE_LINE, name="d0")
    op = Op(kind=OpKind.DRIVE,
            operands=(v_in,),
            results=(v_out,),
            attrs={"duration_vtu": 40})
    assert op.kind == OpKind.DRIVE
    assert duration_of(op) == 40


def test_program_vtu_to_ns():
    prog = Program(name="test",
                   clock_ghz=2.0,
                   ops=[],
                   values=[],
                   qubit_freq_hz={0: 5e9})
    assert prog.vtu_to_ns == 0.5


def test_program_vtu_to_ns_zero_clock():
    prog = Program(name="test",
                   clock_ghz=0.0,
                   ops=[],
                   values=[],
                   qubit_freq_hz={0: 5e9})
    with pytest.raises(ValueError, match="clock_ghz must be positive"):
        _ = prog.vtu_to_ns


def test_clone_program(simple_program):
    cloned = clone_program(simple_program)
    assert cloned.name == simple_program.name
    assert cloned.op_count() == simple_program.op_count()
    assert cloned is not simple_program
    assert cloned.ops is not simple_program.ops


def test_line_id_of():
    d = Value(vid=0, vtype=ValueType.DRIVE_LINE, name="d0")
    op = Op(kind=OpKind.DRIVE, operands=(d,), results=(), attrs={})
    assert line_id_of(op) == 0


def test_tone_id_of():
    t = Value(vid=5, vtype=ValueType.TONE, name="t0")
    op = Op(kind=OpKind.SHIFT_PHASE, operands=(t,), results=(), attrs={})
    assert tone_id_of(op) == 5


def test_is_loop_or_barrier():
    op_for = Op(kind=OpKind.FOR_LOOP, operands=(), results=(), attrs={})
    op_end = Op(kind=OpKind.END_FOR, operands=(), results=(), attrs={})
    op_sync = Op(kind=OpKind.SYNC, operands=(), results=(), attrs={})
    op_drv = Op(kind=OpKind.DRIVE, operands=(), results=(), attrs={})

    assert is_loop_or_barrier(op_for)
    assert is_loop_or_barrier(op_end)
    assert is_loop_or_barrier(op_sync)
    assert not is_loop_or_barrier(op_drv)


def test_mk_helper():
    v = _mk(ValueType.WAVEFORM, "wf")
    assert v.vtype == ValueType.WAVEFORM
    assert v.name == "wf"


def test_reset_vid_counter():
    _reset_vid_counter(999)
    v = _mk(ValueType.TONE, "t")
    assert v.vid == 999
