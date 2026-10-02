# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU integration test: 1-qubit Rabi oscillation.

Verifies the MLIR emission for a Rabi experiment is structurally correct.
Full GPU execution test is marked with @pytest.mark.gpu.
"""

import math

import numpy as np
import pytest

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.passes.to_pulse_mlir import program_to_pulse_mlir
from cudaq_pulse.passes import run_canonicalize, run_virtual_z, run_fusion, schedule_alap
from cudaq_pulse.targets import Qubit, Target


def _gpu_available():
    try:
        from cudaq_pulse.runtime.jit import _check_gpu_available

        return _check_gpu_available()
    except Exception:
        return False


gpu = pytest.mark.gpu
requires_gpu = pytest.mark.skipif(not _gpu_available(),
                                  reason="No GPU/cuDensityMat")


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
    assert "pulse.gaussian" in mlir
    assert "!pulse.waveform" in mlir


@gpu
@requires_gpu
def test_rabi_gpu_execution():
    """Single-qubit Rabi oscillation on GPU."""

    target = Target(
        name="unitary-rabi-test",
        qubits={
            0:
                Qubit(
                    index=0,
                    frequency_hz=5.0e9,
                    anharmonicity_hz=-200.0e6,
                    t1_us=0.0,
                    t2_star_us=0.0,
                    drive_params={"amplitude_scale_rad_per_ns": 1.0},
                )
        },
    )

    @pulse.kernel
    def rabi(q0):
        d0, t0 = get_drive_line(q0)
        wf = square(40, math.pi / 20.0)
        drive(d0, wf, t0)

    ir = rabi(pulse.qudit_ref())
    result = pulse.evolve(ir,
                          target=target,
                          t_start=0.0,
                          t_end=20.0,
                          num_steps=200,
                          integrator="rk4")

    state = result.final_state
    assert state.shape == (2,)
    assert np.vdot(state, state).real == pytest.approx(1.0, abs=1.0e-6)
    assert abs(state[1])**2 > 0.999


@gpu
@requires_gpu
@pytest.mark.parametrize("integrator", ["rk4", "magnus", "crank_nicolson"])
def test_rabi_gpu_integrator_parity(integrator):
    """All cuDensityMat integrators drive the same pi-pulse to |1>.

    Exercises the magnus (Taylor-series midpoint) and crank_nicolson
    (predictor-corrector) paths added to cudm-runtime alongside rk4, and
    confirms they agree on a closed-system Rabi flip while preserving norm.
    """

    target = Target(
        name="unitary-rabi-parity",
        qubits={
            0:
                Qubit(
                    index=0,
                    frequency_hz=5.0e9,
                    anharmonicity_hz=-200.0e6,
                    t1_us=0.0,
                    t2_star_us=0.0,
                    drive_params={"amplitude_scale_rad_per_ns": 1.0},
                )
        },
    )

    @pulse.kernel
    def rabi(q0):
        d0, t0 = get_drive_line(q0)
        wf = square(40, math.pi / 20.0)
        drive(d0, wf, t0)

    ir = rabi(pulse.qudit_ref())
    result = pulse.evolve(ir,
                          target=target,
                          t_start=0.0,
                          t_end=20.0,
                          num_steps=200,
                          integrator=integrator)

    state = result.final_state
    assert state.shape == (2,)
    assert np.vdot(state, state).real == pytest.approx(1.0, abs=1.0e-6)
    assert abs(state[1])**2 > 0.999
