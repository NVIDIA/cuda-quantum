# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU integration tests: T1 amplitude damping and multi-qubit decoherence.

Ported from the research-preview ``qpu/physics/test_decoherence.cpp`` engine
tests to the dialect-routed cuDensityMat runtime. The old parallel-engine
config-sampling cases (``TransmonConfig::generate`` / ``create_ladder_8q``
seeding) are intentionally dropped: parameter sampling is not part of the
pulse frontend. The physics -- T1 decay, monotonicity, ground-state stability,
trace preservation, and independent multi-qubit decay -- is preserved here
through ``Target`` T1 Lindblad terms.
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


def _single_qubit_target(*, t1_us, t2_star_us=0.0, frequency_hz=5.0e9):
    return Target(
        name="decoherence-test",
        qubits={
            0:
                Qubit(
                    index=0,
                    frequency_hz=frequency_hz,
                    anharmonicity_hz=-200.0e6,
                    t1_us=t1_us,
                    t2_star_us=t2_star_us,
                    drive_params={"amplitude_scale_rad_per_ns": 1.0},
                )
        },
    )


# A calibrated pi pulse: 40 virtual units at 2 GHz is 20 ns; with H = amp*X/2
# an amplitude of pi/20 rad/ns applies a pi rotation |0> -> |1>.
def _pi_pulse_then_wait(wait_vtu):

    @pulse.kernel
    def kernel(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0), t0)
        wait(d0, wait_vtu)

    return kernel


def test_decoherence_mlir_structure():
    """A pi-pulse-plus-wait kernel lowers to structurally valid MLIR."""
    ir = _pi_pulse_then_wait(1000)(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    prog = run_canonicalize(prog)
    prog = run_virtual_z(prog)
    prog = run_fusion(prog)
    schedule_alap(prog)
    mlir = program_to_pulse_mlir(prog)

    assert "pulse.square" in mlir
    assert "pulse.wait" in mlir


@gpu
@requires_gpu
def test_t1_excited_state_decays():
    """After a pi pulse and a wait, the excited population damps toward |0>."""
    target = _single_qubit_target(t1_us=0.2)
    ir = _pi_pulse_then_wait(1000)(pulse.qudit_ref())  # 500 ns of free decay
    result = pulse.evolve(ir,
                          target=target,
                          t_start=0.0,
                          t_end=520.0,
                          num_steps=520,
                          integrator="rk4")

    rho = result.final_state
    assert rho.shape == (2, 2)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1.0e-6)
    p1 = rho[1, 1].real
    # exp(-500/200) ~ 0.082; the 20 ns drive adds a little in-pulse decay.
    assert p1 < 0.5
    assert rho[0, 0].real > p1


@gpu
@requires_gpu
def test_t1_longer_wait_more_decay():
    """Longer idle time produces strictly more T1 decay."""
    target = _single_qubit_target(t1_us=0.2)

    def _run(wait_vtu, t_end):
        ir = _pi_pulse_then_wait(wait_vtu)(pulse.qudit_ref())
        result = pulse.evolve(ir,
                              target=target,
                              t_start=0.0,
                              t_end=t_end,
                              num_steps=int(t_end),
                              integrator="rk4")
        return result.final_state[1, 1].real

    p1_short = _run(200, 120.0)  # 100 ns wait
    p1_long = _run(2000, 1020.0)  # 1000 ns wait
    assert p1_long < p1_short
    assert p1_short < 1.0


@gpu
@requires_gpu
def test_ground_state_stable_under_decoherence():
    """The ground state is the fixed point of T1 damping."""
    target = _single_qubit_target(t1_us=0.2)

    @pulse.kernel
    def idle(q0):
        d0, _t0 = get_drive_line(q0)
        wait(d0, 2000)  # 1000 ns of free evolution, no drive

    ir = idle(pulse.qudit_ref())
    result = pulse.evolve(ir,
                          target=target,
                          t_start=0.0,
                          t_end=1000.0,
                          num_steps=1000,
                          integrator="rk4")
    rho = result.final_state
    assert rho.shape == (2, 2)
    assert rho[0, 0].real > 0.98


@gpu
@requires_gpu
def test_decoherence_preserves_trace():
    """Lindblad evolution conserves the density-matrix trace."""
    target = _single_qubit_target(t1_us=0.2)
    ir = _pi_pulse_then_wait(4000)(pulse.qudit_ref())  # 2000 ns
    result = pulse.evolve(ir,
                          target=target,
                          t_start=0.0,
                          t_end=2020.0,
                          num_steps=2020,
                          integrator="rk4")
    rho = result.final_state
    assert np.trace(rho).real == pytest.approx(1.0, abs=1.0e-3)


@gpu
@requires_gpu
def test_multiqubit_independent_decay():
    """Two qubits with distinct T1 both damp; the whole state stays physical."""
    target = Target(
        name="two-qubit-decoherence",
        qubits={
            0:
                Qubit(index=0,
                      frequency_hz=5.0e9,
                      anharmonicity_hz=-200.0e6,
                      t1_us=0.2,
                      t2_star_us=0.0,
                      drive_params={"amplitude_scale_rad_per_ns": 1.0}),
            1:
                Qubit(index=1,
                      frequency_hz=5.2e9,
                      anharmonicity_hz=-200.0e6,
                      t1_us=0.1,
                      t2_star_us=0.0,
                      drive_params={"amplitude_scale_rad_per_ns": 1.0}),
        },
    )

    @pulse.kernel
    def excite_both(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        drive(d0, square(40, math.pi / 20.0), t0)
        drive(d1, square(40, math.pi / 20.0), t1)
        sync(d0, d1)
        wait(d0, 1000)
        wait(d1, 1000)

    ir = excite_both(pulse.qudit_ref(), pulse.qudit_ref())
    result = pulse.evolve(ir,
                          target=target,
                          t_start=0.0,
                          t_end=520.0,
                          num_steps=520,
                          integrator="rk4")

    rho = result.final_state
    assert rho.shape == (4, 4)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1.0e-3)
    # Reduced excited populations: q0 = |01>+|11| diag, q1 = |10>+|11| diag.
    # Basis order is |q0 q1>: indices 0=00,1=01,2=10,3=11.
    p1_q0 = rho[2, 2].real + rho[3, 3].real
    p1_q1 = rho[1, 1].real + rho[3, 3].real
    assert p1_q0 < 0.9
    assert p1_q1 < 0.9
