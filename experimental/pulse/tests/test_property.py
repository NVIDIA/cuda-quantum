# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property-based tests using Hypothesis for the pulse IR."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from cudaq_pulse.passes.ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    _mk,
    _reset_vid_counter,
)
from cudaq_pulse.passes.scheduling import schedule_asap, schedule_alap
from cudaq_pulse.passes.verify import verify
from cudaq_pulse.passes.canonicalize import run_canonicalize
from cudaq_pulse.passes.virtual_z import run_virtual_z
from cudaq_pulse.passes.fusion import run_fusion


def _build_random_program(n_drives: int, n_waits: int, clock_ghz: float,
                          amplitudes: list, durations: list, wait_durs: list):
    """Build a synthetic program with n_drives and n_waits on one line."""
    _reset_vid_counter(10000)
    vid = [10000]

    def nv(vt, nm):
        v = Value(vid=vid[0], vtype=vt, name=nm)
        vid[0] += 1
        return v

    d = nv(ValueType.DRIVE_LINE, "d0")
    t = nv(ValueType.TONE, "t0")
    vals = [d, t]
    ops = [
        Op(kind=OpKind.ALLOC_DRIVE,
           operands=(),
           results=(d, t),
           attrs={
               "qubit": 0,
               "frequency_hz": 5e9
           }),
    ]

    cur_d, cur_t = d, t

    for i in range(n_drives):
        wf = nv(ValueType.WAVEFORM, f"wf{i}")
        d_out = nv(ValueType.DRIVE_LINE, "d0")
        t_out = nv(ValueType.TONE, "t0")
        vals.extend([wf, d_out, t_out])

        dur = durations[i % len(durations)]
        amp = amplitudes[i % len(amplitudes)]
        ops.append(
            Op(kind=OpKind.MAKE_WAVEFORM,
               operands=(),
               results=(wf,),
               attrs={
                   "waveform_type": "gaussian",
                   "duration_vtu": dur,
                   "amplitude": amp,
                   "sigma": max(dur / 4.0, 1.0)
               }))
        ops.append(
            Op(kind=OpKind.DRIVE,
               operands=(cur_d, wf, cur_t),
               results=(d_out, t_out),
               attrs={"duration_vtu": dur}))
        cur_d, cur_t = d_out, t_out

    for i in range(n_waits):
        d_out = nv(ValueType.DRIVE_LINE, "d0")
        vals.append(d_out)
        w = wait_durs[i % len(wait_durs)]
        ops.append(
            Op(kind=OpKind.WAIT,
               operands=(cur_d,),
               results=(d_out,),
               attrs={"duration_vtu": w}))
        cur_d = d_out

    return Program(name="fuzz",
                   clock_ghz=clock_ghz,
                   ops=ops,
                   values=vals,
                   qubit_freq_hz={0: 5e9})


@given(
    n_drives=st.integers(min_value=1, max_value=10),
    n_waits=st.integers(min_value=0, max_value=3),
    clock_ghz=st.floats(min_value=0.1, max_value=10.0),
    amplitudes=st.lists(st.floats(min_value=-1.0,
                                  max_value=1.0,
                                  allow_nan=False,
                                  allow_infinity=False),
                        min_size=1,
                        max_size=5),
    durations=st.lists(st.integers(min_value=4, max_value=1000),
                       min_size=1,
                       max_size=5),
    wait_durs=st.lists(st.integers(min_value=0, max_value=500),
                       min_size=1,
                       max_size=3),
)
@settings(max_examples=50, deadline=5000)
def test_schedule_never_crashes(n_drives, n_waits, clock_ghz, amplitudes,
                                durations, wait_durs):
    """ASAP scheduling should never crash on any valid program."""
    prog = _build_random_program(n_drives, n_waits, clock_ghz, amplitudes,
                                 durations, wait_durs)
    events, metrics = schedule_asap(prog)
    assert metrics.total_length_vtu >= 0


@given(
    n_drives=st.integers(min_value=1, max_value=8),
    clock_ghz=st.floats(min_value=0.5, max_value=5.0),
    amplitudes=st.lists(st.floats(min_value=-1.0,
                                  max_value=1.0,
                                  allow_nan=False,
                                  allow_infinity=False),
                        min_size=1,
                        max_size=4),
    durations=st.lists(st.integers(min_value=4, max_value=500),
                       min_size=1,
                       max_size=4),
)
@settings(max_examples=50, deadline=5000)
def test_asap_alap_same_makespan(n_drives, clock_ghz, amplitudes, durations):
    """ASAP and ALAP should produce the same total length on single-line programs."""
    prog = _build_random_program(n_drives, 0, clock_ghz, amplitudes, durations,
                                 [])
    _, asap_m = schedule_asap(prog)
    _, alap_m = schedule_alap(prog)
    assert abs(asap_m.total_length_vtu - alap_m.total_length_vtu) < 1e-6


@given(
    n_drives=st.integers(min_value=1, max_value=6),
    amplitudes=st.lists(st.floats(min_value=-1.0,
                                  max_value=1.0,
                                  allow_nan=False,
                                  allow_infinity=False),
                        min_size=1,
                        max_size=3),
    durations=st.lists(st.integers(min_value=4, max_value=200),
                       min_size=1,
                       max_size=3),
)
@settings(max_examples=30, deadline=5000)
def test_canonicalize_preserves_drive_count(n_drives, amplitudes, durations):
    """Canonicalize should not drop any DRIVE ops."""
    prog = _build_random_program(n_drives, 0, 2.0, amplitudes, durations, [])
    result = run_canonicalize(prog)
    orig_drives = sum(1 for op in prog.ops if op.kind == OpKind.DRIVE)
    new_drives = sum(1 for op in result.ops if op.kind == OpKind.DRIVE)
    assert new_drives == orig_drives


@given(
    n_drives=st.integers(min_value=1, max_value=5),
    durations=st.lists(st.integers(min_value=4, max_value=200),
                       min_size=1,
                       max_size=3),
)
@settings(max_examples=30, deadline=5000)
def test_virtual_z_idempotent(n_drives, durations):
    """Running virtual_z twice should be the same as running it once."""
    prog = _build_random_program(n_drives, 0, 2.0, [0.3], durations, [])
    once = run_virtual_z(prog)
    twice = run_virtual_z(once)
    assert once.op_count() == twice.op_count()
