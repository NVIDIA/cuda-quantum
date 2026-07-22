# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.passes.ir_types import OpKind


def test_lower_basic_kernel():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf, t0)

    ir = k(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    kinds = [op.kind for op in prog.ops]
    assert OpKind.ALLOC_DRIVE in kinds
    assert OpKind.MAKE_WAVEFORM in kinds
    assert OpKind.DRIVE in kinds


def test_lower_preserves_frequency():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf, t0)

    ir = k(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 4.8e9})
    alloc_ops = [op for op in prog.ops if op.kind == OpKind.ALLOC_DRIVE]
    assert len(alloc_ops) >= 1
    assert alloc_ops[0].attrs.get("frequency_hz") == 4.8e9


def test_lower_missing_frequency_raises():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf, t0)

    ir = k(pulse.qudit_ref())
    from cudaq_pulse.kernel.ir_builder import CompilationError
    with pytest.raises(CompilationError, match="no frequency provided"):
        to_program(ir, clock_ghz=2.0, qubit_freq_hz={})


def test_lower_wait_duration():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        wait(d0, 100)

    ir = k(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5e9})
    wait_ops = [op for op in prog.ops if op.kind == OpKind.WAIT]
    assert len(wait_ops) >= 1
    assert wait_ops[0].attrs.get("duration_vtu") == 100
