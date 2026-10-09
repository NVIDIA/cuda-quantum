# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU integration test: T1 exponential decay."""

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

    assert "pulse.gaussian" in mlir
    assert "pulse.wait" in mlir
    assert "arith.constant 1000 : i64" in mlir


@gpu
@requires_gpu
def test_t1_decay_gpu():
    """T1 decay: after exciting to |1> and waiting, population decays."""

    target = Target(
        name="fast-decay-test",
        qubits={
            0:
                Qubit(
                    index=0,
                    frequency_hz=5.0e9,
                    anharmonicity_hz=-200.0e6,
                    t1_us=0.05,
                    t2_star_us=0.0,
                    drive_params={"amplitude_scale_rad_per_ns": 1.0},
                )
        },
    )

    @pulse.kernel
    def t1_decay(q0):
        d0, t0 = get_drive_line(q0)
        # 40 virtual units at 2 GHz is 20 ns. With H = amplitude * X / 2,
        # amplitude pi/20 applies a pi rotation.
        pi_pulse = square(40, math.pi / 20.0)
        drive(d0, pi_pulse, t0)
        wait(d0, 1000)

    ir = t1_decay(pulse.qudit_ref())
    result = pulse.evolve(ir,
                          target=target,
                          t_start=0.0,
                          t_end=520.0,
                          num_steps=520,
                          integrator="rk4")

    state = result.final_state
    assert state.shape == (2, 2)
    assert np.trace(state) == pytest.approx(1.0, abs=1.0e-6)
    assert state[1, 1].real < 1.0e-3
