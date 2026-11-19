# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU integration test: 2-qubit Bell state from XX coupling."""

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
@requires_gpu
def test_bell_gpu_execution():
    """XX evolution produces (|00> - i|11>) / sqrt(2)."""

    qubits = {
        index:
            Qubit(
                index=index,
                frequency_hz=(5.0 + 0.1 * index) * 1.0e9,
                anharmonicity_hz=-200.0e6,
                t1_us=0.0,
                t2_star_us=0.0,
            ) for index in range(2)
    }
    target = Target(
        name="xx-bell-test",
        qubits=qubits,
        couplings=[Coupling(0, 1, coupling_strength_hz=5.0e6)],
    )

    @pulse.kernel
    def bell_xx(q0, q1):
        d0, _t0 = get_drive_line(q0)
        d1, _t1 = get_drive_line(q1)
        wait(d0, 50)
        wait(d1, 50)
        sync(d0, d1)

    ir = bell_xx(pulse.qudit_ref(), pulse.qudit_ref())
    result = pulse.evolve(ir,
                          target=target,
                          t_start=0.0,
                          t_end=25.0,
                          num_steps=250,
                          integrator="rk4")

    state = result.final_state
    expected = np.array([1.0, 0.0, 0.0, -1.0j]) / math.sqrt(2.0)
    fidelity = abs(np.vdot(expected, state))**2
    assert state.shape == (4,)
    assert fidelity > 0.999
