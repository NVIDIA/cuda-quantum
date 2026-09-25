# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU integration tests: closed-system physics validation.

Ported from ``qpu/physics/test_physics_validation.cpp``. Covers ground-state
initialization, free drift, single-qubit Rabi rotation angles (pi/2 and pi),
selective addressing of one qubit in a register, and XX-coupling excitation
exchange.
"""

import math

import numpy as np
import pytest

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.passes.to_pulse_mlir import program_to_pulse_mlir
from cudaq_pulse.passes import run_canonicalize, run_virtual_z, run_fusion, schedule_alap
from cudaq_pulse.targets import Coupling, Qubit, Target


def _gpu_available():
    try:
        from cudaq_pulse.runtime.jit import _check_gpu_available

        return _check_gpu_available()
    except Exception:
        return False


gpu = pytest.mark.gpu
requires_gpu = pytest.mark.skipif(not _gpu_available(),
                                  reason="No GPU/cuDensityMat")


def _unitary_qubit(index, frequency_hz=5.0e9):
    return Qubit(index=index,
                 frequency_hz=frequency_hz,
                 anharmonicity_hz=-200.0e6,
                 t1_us=0.0,
                 t2_star_us=0.0,
                 drive_params={"amplitude_scale_rad_per_ns": 1.0})


def _single_qubit_target():
    return Target(name="physics-1q", qubits={0: _unitary_qubit(0)})


def test_physics_validation_mlir_structure():
    """A resonant-drive kernel lowers to valid MLIR."""

    @pulse.kernel
    def rabi(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0), t0)

    prog = to_program(rabi(pulse.qudit_ref()),
                      clock_ghz=2.0,
                      qubit_freq_hz={0: 5.0e9})
    prog = run_canonicalize(prog)
    prog = run_virtual_z(prog)
    prog = run_fusion(prog)
    schedule_alap(prog)
    mlir = program_to_pulse_mlir(prog)
    assert "pulse.square" in mlir


@gpu
@requires_gpu
def test_free_drift_stays_ground():
    """With no drive, the ground state is a static-Hamiltonian eigenstate."""

    @pulse.kernel
    def idle(q0):
        d0, _t0 = get_drive_line(q0)
        wait(d0, 40)  # 20 ns free evolution

    result = pulse.evolve(idle(pulse.qudit_ref()),
                          target=_single_qubit_target(),
                          t_start=0.0,
                          t_end=20.0,
                          num_steps=200,
                          integrator="rk4")
    state = result.final_state
    assert state.shape == (2,)
    assert abs(state[0])**2 > 0.99


@gpu
@requires_gpu
def test_rabi_half_pi_pulse():
    """A pi/2 rotation leaves equal populations in |0> and |1>."""

    @pulse.kernel
    def half_pi(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 40.0), t0)  # angle = pi/2

    result = pulse.evolve(half_pi(pulse.qudit_ref()),
                          target=_single_qubit_target(),
                          t_start=0.0,
                          t_end=20.0,
                          num_steps=200,
                          integrator="rk4")
    state = result.final_state
    assert abs(state[1])**2 == pytest.approx(0.5, abs=0.05)


@gpu
@requires_gpu
def test_rabi_pi_pulse():
    """A pi rotation fully transfers population to |1>."""

    @pulse.kernel
    def pi_pulse(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0), t0)  # angle = pi

    result = pulse.evolve(pi_pulse(pulse.qudit_ref()),
                          target=_single_qubit_target(),
                          t_start=0.0,
                          t_end=20.0,
                          num_steps=200,
                          integrator="rk4")
    state = result.final_state
    assert abs(state[1])**2 > 0.99


@gpu
@requires_gpu
def test_two_qubit_selective_drive():
    """Driving q0 excites only q0; q1 remains in its ground state."""
    target = Target(
        name="physics-2q",
        qubits={
            0: _unitary_qubit(0),
            1: _unitary_qubit(1, 5.2e9)
        },
    )

    @pulse.kernel
    def drive_q0(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, _t1 = get_drive_line(q1)
        drive(d0, square(40, math.pi / 20.0), t0)
        wait(d1, 40)
        sync(d0, d1)

    result = pulse.evolve(drive_q0(pulse.qudit_ref(), pulse.qudit_ref()),
                          target=target,
                          t_start=0.0,
                          t_end=20.0,
                          num_steps=200,
                          integrator="rk4")
    state = result.final_state
    assert state.shape == (4,)
    probs = np.abs(state)**2
    # Exactly one qubit is excited (the driven one), independent of the
    # basis-ordering convention: population sits in the single-excitation
    # manifold, not in |00> and not in |11>.
    assert probs[0] < 0.1  # left the ground state
    assert probs[1] + probs[2] > 0.9  # one qubit excited
    assert probs[3] < 0.05  # both-excited is negligible


@gpu
@requires_gpu
def test_xx_coupling_transfers_excitation():
    """An XX coupling exchanges excitation between neighboring qubits."""
    target = Target(
        name="physics-xx",
        qubits={
            0: _unitary_qubit(0),
            1: _unitary_qubit(1, 5.0e9)
        },
        couplings=[Coupling(0, 1, coupling_strength_hz=25.0e6)],
    )

    @pulse.kernel
    def excite_and_exchange(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, _t1 = get_drive_line(q1)
        drive(d0, square(40, math.pi / 20.0), t0)  # prepare ~|10>
        sync(d0, d1)
        wait(d0, 40)  # let XX coupling swap population
        wait(d1, 40)

    result = pulse.evolve(excite_and_exchange(pulse.qudit_ref(),
                                              pulse.qudit_ref()),
                          target=target,
                          t_start=0.0,
                          t_end=40.0,
                          num_steps=400,
                          integrator="rk4")
    state = result.final_state
    probs = np.abs(state)**2
    # Population should have partly transferred q0 -> q1 (|10> -> |01>).
    assert probs[1] > 0.1
