# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU integration tests: quantum-algorithm building blocks.

Ported from ``qpu/physics/test_quantum_algorithms.cpp`` (whose entangling-gate
cases were all ``DISABLED_`` upstream because pulse-calibrated Bell/GHZ/CNOT
fidelity was not achievable on the parallel engine). Here the robust
single-qubit rotations and gate-sequence identities run on GPU, while the
multi-qubit entangling schedules are exercised as compile/lowering structure
tests rather than fidelity assertions.
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


def _unitary_qubit(index, frequency_hz):
    return Qubit(index=index,
                 frequency_hz=frequency_hz,
                 anharmonicity_hz=-200.0e6,
                 t1_us=0.0,
                 t2_star_us=0.0,
                 drive_params={"amplitude_scale_rad_per_ns": 1.0})


def _single_qubit_target():
    return Target(name="algo-1q", qubits={0: _unitary_qubit(0, 5.0e9)})


def _evolve_1q(kernel):
    return pulse.evolve(kernel(pulse.qudit_ref()),
                        target=_single_qubit_target(),
                        t_start=0.0,
                        t_end=40.0,
                        num_steps=400,
                        integrator="rk4").final_state


def test_bell_schedule_mlir_structure():
    """A CR-based Bell schedule lowers to valid two-qubit MLIR."""

    @pulse.kernel
    def bell(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        shift_phase(t0, math.pi / 2)
        drive(d0, drag(40, 0.25, 10.0, 0.5), t0)
        shift_phase(t0, math.pi / 2)
        sync(d0, d1)
        drive(d0, gaussian(160, 0.05, 40.0), t1)  # cross-resonance
        sync(d0, d1)

    prog = to_program(bell(pulse.qudit_ref(), pulse.qudit_ref()),
                      clock_ghz=2.0,
                      qubit_freq_hz={
                          0: 5.0e9,
                          1: 5.1e9
                      })
    mlir = program_to_pulse_mlir(prog)
    assert mlir.count("pulse.qudit_alloc") == 2
    assert "pulse.sync" in mlir


def test_ghz_schedule_mlir_structure():
    """A 3-qubit GHZ-style chain schedule lowers to valid MLIR."""

    @pulse.kernel
    def ghz(q0, q1, q2):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        d2, t2 = get_drive_line(q2)
        drive(d0, square(40, math.pi / 40.0), t0)
        sync(d0, d1, d2)
        drive(d0, gaussian(160, 0.05, 40.0), t1)
        drive(d1, gaussian(160, 0.05, 40.0), t2)
        sync(d0, d1, d2)

    prog = to_program(ghz(pulse.qudit_ref(), pulse.qudit_ref(),
                          pulse.qudit_ref()),
                      clock_ghz=2.0,
                      qubit_freq_hz={
                          0: 5.0e9,
                          1: 5.1e9,
                          2: 5.2e9
                      })
    mlir = program_to_pulse_mlir(prog)
    assert mlir.count("pulse.qudit_alloc") == 3


@gpu
@requires_gpu
def test_single_qubit_x_rotation():
    """An X(pi) rotation flips |0> -> |1>."""

    @pulse.kernel
    def x_pi(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0), t0)

    assert abs(_evolve_1q(x_pi)[1])**2 > 0.99


@gpu
@requires_gpu
def test_single_qubit_y_rotation():
    """A Y(pi) rotation (phase pi/2) flips |0> -> |1>."""

    @pulse.kernel
    def y_pi(q0):
        d0, t0 = get_drive_line(q0)
        shift_phase(t0, math.pi / 2)
        drive(d0, square(40, math.pi / 20.0), t0)

    assert abs(_evolve_1q(y_pi)[1])**2 > 0.99


@gpu
@requires_gpu
def test_bloch_half_rotation():
    """An X(pi/2) rotation produces an equal superposition."""

    @pulse.kernel
    def x90(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 40.0), t0)

    assert abs(_evolve_1q(x90)[1])**2 == pytest.approx(0.5, abs=0.05)


@gpu
@requires_gpu
def test_gate_sequence_identity():
    """Two X(pi) rotations compose to the identity, returning to |0>."""

    @pulse.kernel
    def xx(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0), t0)
        drive(d0, square(40, math.pi / 20.0), t0)

    assert abs(_evolve_1q(xx)[0])**2 > 0.95
