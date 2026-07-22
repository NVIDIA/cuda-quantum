# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudaq_pulse.passes.ir_types import Op, OpKind, Program, Value, ValueType, _mk, _reset_vid_counter
from cudaq_pulse.passes.verify import (
    verify,
    LinearityViolation,
    BackwardTimeTravelError,
    PhaseBookkeepingError,
)


def test_valid_program(simple_program):
    issues = verify(simple_program)
    errors = [i for i in issues if i.severity == "error"]
    linearity_unconsumed = [i for i in errors if "never consumed" in i.message]
    non_linearity = [i for i in errors if "never consumed" not in i.message]
    assert len(non_linearity) == 0


def test_valid_two_qubit(two_qubit_program):
    issues = verify(two_qubit_program)
    errors = [i for i in issues if i.severity == "error"]
    non_linearity = [i for i in errors if "never consumed" not in i.message]
    assert len(non_linearity) == 0


def test_valid_echo(echo_program):
    issues = verify(echo_program)
    errors = [i for i in issues if i.severity == "error"]
    non_linearity = [i for i in errors if "never consumed" not in i.message]
    assert len(non_linearity) == 0


def test_double_use_line():
    """Two drives consuming the same line value -> linearity violation."""
    _reset_vid_counter(100)
    d0 = Value(vid=100, vtype=ValueType.DRIVE_LINE, name="d0")
    t0 = Value(vid=101, vtype=ValueType.TONE, name="t0")
    wf = Value(vid=102, vtype=ValueType.WAVEFORM, name="wf")
    d0_out1 = Value(vid=103, vtype=ValueType.DRIVE_LINE, name="d0")
    t0_out1 = Value(vid=104, vtype=ValueType.TONE, name="t0")
    d0_out2 = Value(vid=105, vtype=ValueType.DRIVE_LINE, name="d0")
    t0_out2 = Value(vid=106, vtype=ValueType.TONE, name="t0")

    p = Program(
        name="bad_linearity",
        clock_ghz=2.0,
        ops=[
            Op(kind=OpKind.ALLOC_DRIVE,
               operands=(),
               results=(d0, t0),
               attrs={
                   "qubit": 0,
                   "frequency_hz": 5e9
               }),
            Op(kind=OpKind.MAKE_WAVEFORM,
               operands=(),
               results=(wf,),
               attrs={
                   "waveform_type": "gaussian",
                   "duration_vtu": 40,
                   "amplitude": 0.3
               }),
            Op(kind=OpKind.DRIVE,
               operands=(d0, wf, t0),
               results=(d0_out1, t0_out1),
               attrs={"duration_vtu": 40}),
            Op(kind=OpKind.DRIVE,
               operands=(d0, wf, t0),
               results=(d0_out2, t0_out2),
               attrs={"duration_vtu": 40}),
        ],
        values=[d0, t0, wf, d0_out1, t0_out1, d0_out2, t0_out2],
        qubit_freq_hz={0: 5e9},
    )
    issues = verify(p)
    assert any(isinstance(i, LinearityViolation) for i in issues)


def test_negative_wait():
    """Scheduled ops with backward start_vtu trigger BackwardTimeTravelError."""
    d0 = Value(vid=300, vtype=ValueType.DRIVE_LINE, name="d0")
    t0 = Value(vid=301, vtype=ValueType.TONE, name="t0")
    wf = Value(vid=302, vtype=ValueType.WAVEFORM, name="wf")
    d0_a = Value(vid=303, vtype=ValueType.DRIVE_LINE, name="d0")
    t0_a = Value(vid=304, vtype=ValueType.TONE, name="t0")
    d0_b = Value(vid=305, vtype=ValueType.DRIVE_LINE, name="d0")
    t0_b = Value(vid=306, vtype=ValueType.TONE, name="t0")

    p = Program(
        name="bad_time",
        clock_ghz=2.0,
        ops=[
            Op(kind=OpKind.ALLOC_DRIVE,
               operands=(),
               results=(d0, t0),
               attrs={
                   "qubit": 0,
                   "frequency_hz": 5e9
               }),
            Op(kind=OpKind.MAKE_WAVEFORM,
               operands=(),
               results=(wf,),
               attrs={
                   "waveform_type": "gaussian",
                   "duration_vtu": 40,
                   "amplitude": 0.3
               }),
            Op(kind=OpKind.DRIVE,
               operands=(d0, wf, t0),
               results=(d0_a, t0_a),
               attrs={
                   "duration_vtu": 40,
                   "start_vtu": 0
               }),
            Op(kind=OpKind.DRIVE,
               operands=(d0_a, wf, t0_a),
               results=(d0_b, t0_b),
               attrs={
                   "duration_vtu": 40,
                   "start_vtu": 10
               }),
        ],
        values=[d0, t0, wf, d0_a, t0_a, d0_b, t0_b],
        qubit_freq_hz={0: 5e9},
    )
    issues = verify(p)
    assert any(isinstance(i, BackwardTimeTravelError) for i in issues)
