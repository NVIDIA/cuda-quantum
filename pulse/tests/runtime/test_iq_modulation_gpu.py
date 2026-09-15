# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU integration tests: I/Q quadrature modulation.

Ported (partially) from ``qpu/physics/test_iq_modulation.cpp``. The in-phase
(I) quadrature drives an X rotation; the quadrature (Q) component -- realized
here as a pi/2 frame phase shift before the drive -- drives a Y rotation. The
pulse->operator lowering models these as amplitude*cos(phase) (X) and
amplitude*sin(phase) (Y) control terms.
"""

import math

import numpy as np
import pytest

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.passes.to_pulse_mlir import program_to_pulse_mlir
from cudaq_pulse.passes import run_canonicalize, run_fusion, schedule_alap
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


def _target():
    return Target(
        name="iq-modulation",
        qubits={
            0:
                Qubit(index=0,
                      frequency_hz=5.0e9,
                      anharmonicity_hz=-200.0e6,
                      t1_us=0.0,
                      t2_star_us=0.0,
                      drive_params={"amplitude_scale_rad_per_ns": 1.0})
        },
    )


def _evolve(kernel):
    return pulse.evolve(kernel(pulse.qudit_ref()),
                        target=_target(),
                        t_start=0.0,
                        t_end=20.0,
                        num_steps=200,
                        integrator="rk4").final_state


def test_iq_modulation_mlir_structure():
    """A quadrature (phase-shifted) drive lowers to valid MLIR."""

    @pulse.kernel
    def q_drive(q0):
        d0, t0 = get_drive_line(q0)
        shift_phase(t0, math.pi / 2)
        drive(d0, square(40, math.pi / 20.0), t0)

    prog = to_program(q_drive(pulse.qudit_ref()),
                      clock_ghz=2.0,
                      qubit_freq_hz={0: 5.0e9})
    prog = run_canonicalize(prog)
    prog = run_fusion(prog)
    schedule_alap(prog)
    mlir = program_to_pulse_mlir(prog)
    assert "pulse.square" in mlir


@gpu
@requires_gpu
def test_pure_i_drives_x_rotation():
    """In-phase (phase 0) pi drive flips |0> -> |1>."""

    @pulse.kernel
    def x_pi(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0), t0)

    assert abs(_evolve(x_pi)[1])**2 > 0.99


@gpu
@requires_gpu
def test_pure_q_drives_y_rotation():
    """Quadrature (phase pi/2) pi drive also flips |0> -> |1>."""

    @pulse.kernel
    def y_pi(q0):
        d0, t0 = get_drive_line(q0)
        shift_phase(t0, math.pi / 2)
        drive(d0, square(40, math.pi / 20.0), t0)

    assert abs(_evolve(y_pi)[1])**2 > 0.99


@gpu
@requires_gpu
def test_iq_symmetry_equal_populations():
    """Equal-magnitude X and Y pi/2 drives yield equal |1> population."""

    @pulse.kernel
    def x90(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 40.0), t0)

    @pulse.kernel
    def y90(q0):
        d0, t0 = get_drive_line(q0)
        shift_phase(t0, math.pi / 2)
        drive(d0, square(40, math.pi / 40.0), t0)

    p1_x = abs(_evolve(x90)[1])**2
    p1_y = abs(_evolve(y90)[1])**2
    assert p1_x == pytest.approx(0.5, abs=0.05)
    assert p1_y == pytest.approx(p1_x, abs=0.05)


@gpu
@requires_gpu
def test_xy_orthogonality_distinct_states():
    """X(pi/2) and Y(pi/2) produce distinguishable states (different phases)."""

    @pulse.kernel
    def x90(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 40.0), t0)

    @pulse.kernel
    def y90(q0):
        d0, t0 = get_drive_line(q0)
        shift_phase(t0, math.pi / 2)
        drive(d0, square(40, math.pi / 40.0), t0)

    x_state = _evolve(x90)
    y_state = _evolve(y90)
    overlap = abs(np.vdot(x_state, y_state))**2
    # Same populations but different relative phase => not the same state.
    assert overlap < 0.9
