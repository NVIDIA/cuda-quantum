# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.passes.ir_types import OpKind
from cudaq_pulse.passes.virtual_z import run_virtual_z


@pulse.kernel
def _vz_absorb(q0):
    d0, t0 = get_drive_line(q0)
    shift_phase(t0, math.pi / 2)
    wf = gaussian(40, 0.3, 10.0)
    drive(d0, wf, t0)


@pulse.kernel
def _vz_merge(q0):
    d0, t0 = get_drive_line(q0)
    shift_phase(t0, 0.1)
    shift_phase(t0, 0.2)
    wf = gaussian(40, 0.3, 10.0)
    drive(d0, wf, t0)


@pulse.kernel
def _vz_noop(q0):
    d0, t0 = get_drive_line(q0)
    wf = gaussian(40, 0.3, 10.0)
    drive(d0, wf, t0)


def test_shift_phase_absorbed():
    ir = _vz_absorb(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    result = run_virtual_z(prog)
    shift_count = sum(1 for op in result.ops if op.kind == OpKind.SHIFT_PHASE)
    drive_ops = [op for op in result.ops if op.kind == OpKind.DRIVE]
    assert shift_count == 0, f"shift_phase should be absorbed, got {shift_count}"
    assert len(drive_ops) == 1
    assert drive_ops[0].attrs.get("virtual_z_applied") is True
    assert abs(drive_ops[0].attrs["phase"] - math.pi / 2) < 1e-10


def test_consecutive_shifts_merge():
    ir = _vz_merge(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    result = run_virtual_z(prog)
    shift_count = sum(1 for op in result.ops if op.kind == OpKind.SHIFT_PHASE)
    drive_ops = [op for op in result.ops if op.kind == OpKind.DRIVE]
    assert shift_count == 0, "both shifts should merge and absorb into drive"
    assert len(drive_ops) == 1
    assert abs(drive_ops[0].attrs["phase"] - 0.3) < 1e-10


def test_no_phase_no_change():
    ir = _vz_noop(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    result = run_virtual_z(prog)
    drive_ops = [op for op in result.ops if op.kind == OpKind.DRIVE]
    assert all("virtual_z_applied" not in op.attrs for op in drive_ops)
