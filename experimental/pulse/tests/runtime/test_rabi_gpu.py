# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU integration test: 1-qubit Rabi oscillation.

Verifies the MLIR emission for a Rabi experiment is structurally correct.
Full GPU execution test is marked with @pytest.mark.gpu.
"""

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


def test_rabi_mlir_structure():
    """Verify the MLIR text structure for a Rabi simulation."""

    @pulse.kernel
    def rabi(q0):
        d0, t0 = get_drive_line(q0)
        wf = gaussian(100, 0.1, 25.0)
        drive(d0, wf, t0)

    ir = rabi(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    prog = run_canonicalize(prog)
    prog = run_virtual_z(prog)
    prog = run_fusion(prog)
    schedule_alap(prog)
    mlir = program_to_pulse_mlir(prog)

    assert "module @rabi" in mlir
    assert "pulse.qudit_alloc" in mlir
    assert "pulse.get_drive_line" in mlir
    assert "pulse.gaussian 100" in mlir
    assert "!pulse.waveform" in mlir


@gpu
def test_rabi_gpu_execution():
    """Single-qubit Rabi oscillation on GPU."""

    @pulse.kernel
    def rabi(q0):
        d0, t0 = get_drive_line(q0)
        wf = gaussian(100, 0.1, 25.0)
        drive(d0, wf, t0)

    ir = rabi(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    prog = run_canonicalize(prog)
    prog = run_virtual_z(prog)
    prog = run_fusion(prog)
    schedule_alap(prog)
    mlir = program_to_pulse_mlir(prog)

    from cudaq_pulse.runtime.jit import compile_and_run_pulse
    results = compile_and_run_pulse(mlir, n_qubits=1)
    state = results[0].to_numpy()
    assert len(state) == 2
    assert abs(abs(state[0])**2 + abs(state[1])**2 - 1.0) < 0.01
