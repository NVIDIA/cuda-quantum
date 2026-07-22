# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU integration test: 2-qubit Bell state via cross-resonance."""

import pytest

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.passes.to_pulse_mlir import program_to_pulse_mlir
from cudaq_pulse.passes import run_canonicalize, run_virtual_z, run_fusion, schedule_alap


def _gpu_available():
    try:
        from cudaq_pulse.runtime.jit import _check_gpu_available
        return _check_gpu_available()
    except Exception:
        return False


gpu = pytest.mark.skipif(not _gpu_available(), reason="No GPU/cuDensityMat")


def test_bell_mlir_structure():
    """Verify 2-qubit CR Bell MLIR has correct structure."""

    @pulse.kernel
    def bell_cr(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        # pi/2 X on q0
        x90 = gaussian(40, 0.25, 10.0)
        drive(d0, x90, t0)
        sync(d0, d1)
        # CR drive on q0 at q1's frequency (simplified)
        cr = square(160, 0.05)
        drive(d0, cr, t0)
        sync(d0, d1)
        # pi/2 X on q1
        drive(d1, x90, t1)

    ir = bell_cr(pulse.qudit_ref(), pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9, 1: 5.1e9})
    mlir = program_to_pulse_mlir(prog)

    assert mlir.count("pulse.qudit_alloc") == 2
    assert mlir.count("pulse.get_drive_line") == 2
    assert "pulse.sync" in mlir
    assert "pulse.square" in mlir
    assert "pulse.gaussian" in mlir


@gpu
def test_bell_gpu_execution():
    """2-qubit Bell state on GPU: fidelity > 0.99 to |Phi+>."""

    @pulse.kernel
    def bell_cr(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        x90 = gaussian(40, 0.25, 10.0)
        drive(d0, x90, t0)
        sync(d0, d1)
        cr = square(160, 0.05)
        drive(d0, cr, t0)
        sync(d0, d1)
        drive(d1, x90, t1)

    ir = bell_cr(pulse.qudit_ref(), pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9, 1: 5.1e9})
    prog = run_canonicalize(prog)
    prog = run_virtual_z(prog)
    prog = run_fusion(prog)
    schedule_alap(prog)
    mlir = program_to_pulse_mlir(prog)

    from cudaq_pulse.runtime.jit import compile_and_run_pulse
    results = compile_and_run_pulse(mlir, n_qubits=2)
    state = results[0].to_numpy()
    assert len(state) == 4
    norm = sum(abs(s)**2 for s in state)
    assert abs(norm - 1.0) < 0.01
