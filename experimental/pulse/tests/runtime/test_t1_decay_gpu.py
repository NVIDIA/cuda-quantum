# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU integration test: T1 exponential decay."""

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


def test_t1_mlir_structure():
    """Verify T1 decay kernel produces valid MLIR."""

    @pulse.kernel
    def t1_decay(q0):
        d0, t0 = get_drive_line(q0)
        # pi pulse to |1>
        pi_pulse = gaussian(40, 0.5, 10.0)
        drive(d0, pi_pulse, t0)
        # wait for T1 decay
        wait(d0, 1000)

    ir = t1_decay(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    prog = run_canonicalize(prog)
    prog = run_virtual_z(prog)
    prog = run_fusion(prog)
    schedule_alap(prog)
    mlir = program_to_pulse_mlir(prog)

    assert "pulse.gaussian 40" in mlir
    assert "pulse.wait" in mlir
    assert "arith.constant 1000 : i64" in mlir


@gpu
def test_t1_decay_gpu():
    """T1 decay: after exciting to |1> and waiting, population decays."""

    @pulse.kernel
    def t1_decay(q0):
        d0, t0 = get_drive_line(q0)
        pi_pulse = gaussian(40, 0.5, 10.0)
        drive(d0, pi_pulse, t0)
        wait(d0, 1000)

    ir = t1_decay(pulse.qudit_ref())
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
