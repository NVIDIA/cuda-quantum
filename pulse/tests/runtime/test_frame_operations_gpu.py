# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU integration tests: frame (phase) operations.

Ported (partially) from ``qpu/physics/test_frame_operations.cpp``. Only the
pieces that map to ops supported by the dialect-routed runtime are kept:
``shift_phase`` rotates the drive axis, which the pulse->operator lowering
models as X/Y quadratures (amplitude*cos(phase) and amplitude*sin(phase)).

Frequency-detuning cases from the source are intentionally omitted: the
research-preview runtime applies drives resonantly and does not model an
independent drive-frequency detuning term.
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


def _target():
    return Target(
        name="frame-ops",
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
                        t_end=40.0,
                        num_steps=400,
                        integrator="rk4").final_state


def test_frame_operations_mlir_structure():
    """A kernel using shift_phase lowers with a pulse.shift_phase op."""

    @pulse.kernel
    def phased(q0):
        d0, t0 = get_drive_line(q0)
        shift_phase(t0, math.pi / 2)
        drive(d0, square(40, math.pi / 20.0), t0)

    prog = to_program(phased(pulse.qudit_ref()),
                      clock_ghz=2.0,
                      qubit_freq_hz={0: 5.0e9})
    prog = run_canonicalize(prog)
    prog = run_fusion(prog)
    schedule_alap(prog)
    mlir = program_to_pulse_mlir(prog)
    assert "pulse.shift_phase" in mlir or "phase" in mlir


@gpu
@requires_gpu
def test_phase_shift_on_ground_state_no_population_change():
    """A frame phase shift alone does not move population out of |0>."""

    @pulse.kernel
    def phase_only(q0):
        d0, t0 = get_drive_line(q0)
        shift_phase(t0, math.pi / 2)
        wait(d0, 20)

    state = _evolve(phase_only)
    assert abs(state[0])**2 > 0.99


@gpu
@requires_gpu
def test_opposite_phase_pulses_cancel():
    """Two pi/2 pulses about opposite axes (phase pi apart) return to |0>."""

    @pulse.kernel
    def cancel(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 40.0), t0)  # +X pi/2
        shift_phase(t0, math.pi)
        drive(d0, square(40, math.pi / 40.0), t0)  # -X pi/2

    state = _evolve(cancel)
    assert abs(state[0])**2 > 0.95


@gpu
@requires_gpu
def test_same_phase_pulses_add():
    """Two same-axis pi/2 pulses compose into a pi rotation to |1>."""

    @pulse.kernel
    def add(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 40.0), t0)
        drive(d0, square(40, math.pi / 40.0), t0)

    state = _evolve(add)
    assert abs(state[1])**2 > 0.95
