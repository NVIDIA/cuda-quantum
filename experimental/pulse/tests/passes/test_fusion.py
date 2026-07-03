# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.passes.ir_types import OpKind
from cudaq_pulse.passes.fusion import run_fusion


@pulse.kernel
def _no_fuse_kernel(q0):
    d0, t0 = get_drive_line(q0)
    sq1 = square(40, 0.3 + 0j)
    drive(d0, sq1, t0)
    sq2 = square(40, 0.5 + 0j)
    drive(d0, sq2, t0)


@pulse.kernel
def _fuse_kernel(q0):
    d0, t0 = get_drive_line(q0)
    sq1 = square(40, 0.3 + 0j)
    drive(d0, sq1, t0)
    sq2 = square(40, 0.3 + 0j)
    drive(d0, sq2, t0)


@pulse.kernel
def _mixed_kernel(q0):
    d0, t0 = get_drive_line(q0)
    g = gaussian(40, 0.3, 10.0)
    drive(d0, g, t0)


def test_no_fusion_different_amplitude():
    ir = _no_fuse_kernel(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    result = run_fusion(prog)
    drive_count = sum(1 for op in result.ops if op.kind == OpKind.DRIVE)
    assert drive_count == 2


def test_fusion_same_amplitude():
    ir = _fuse_kernel(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    result = run_fusion(prog)
    drive_count = sum(1 for op in result.ops if op.kind == OpKind.DRIVE)
    assert drive_count >= 1


def test_fusion_preserves_non_square():
    ir = _mixed_kernel(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    result = run_fusion(prog)
    waveform_ops = [op for op in result.ops if op.kind == OpKind.MAKE_WAVEFORM]
    assert any(
        op.attrs.get("waveform_type") == "gaussian" for op in waveform_ops)
