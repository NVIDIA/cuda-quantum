# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU integration tests: 8-qubit ladder system.

Ported from ``qpu/physics/test_8qubit_system.cpp``. The engine-specific
calibration/dispersive-readout machinery (``SystemCalibration``,
``TransmonConfig::create_ladder_8q``, MLIR calibration files) is dropped;
what survives is the ladder connectivity as a ``Target`` and a scalable
256-dimensional GPU evolution.

    q0 -- q1 -- q2 -- q3
    |     |     |     |
    q4 -- q5 -- q6 -- q7
"""

import math

import numpy as np
import pytest

import cudaq_pulse as pulse
from cudaq_pulse.targets import Coupling, Qubit, Target

_HORIZONTAL = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]
_VERTICAL = [(0, 4), (1, 5), (2, 6), (3, 7)]
_LADDER_EDGES = _HORIZONTAL + _VERTICAL


def _gpu_available():
    try:
        from cudaq_pulse.runtime.jit import _check_gpu_available

        return _check_gpu_available()
    except Exception:
        return False


gpu = pytest.mark.gpu
requires_gpu = pytest.mark.skipif(not _gpu_available(),
                                  reason="No GPU/cuDensityMat")


def _ladder_8q_target(coupling_hz=1.0e6):
    qubits = {
        i:
            Qubit(index=i,
                  frequency_hz=(5.0 + 0.02 * i) * 1.0e9,
                  anharmonicity_hz=-200.0e6,
                  t1_us=0.0,
                  t2_star_us=0.0,
                  drive_params={"amplitude_scale_rad_per_ns": 1.0})
        for i in range(8)
    }
    couplings = [
        Coupling(a, b, coupling_strength_hz=coupling_hz)
        for a, b in _LADDER_EDGES
    ]
    return Target(name="ladder-8q", qubits=qubits, couplings=couplings)


def test_ladder_topology_structure():
    """The ladder target encodes the expected 10-edge connectivity."""
    target = _ladder_8q_target()
    assert len(target.couplings) == 10

    edges = {tuple(sorted(pair)) for pair in target.coupling_map}
    assert (0, 1) in edges
    assert (0, 4) in edges
    assert (5, 6) in edges
    assert (3, 7) in edges
    # Non-adjacent qubits are not directly coupled.
    assert (0, 2) not in edges
    assert (0, 7) not in edges
    assert (4, 7) not in edges


def test_ladder_neighbor_degrees():
    """Corner qubits have degree 2; edge/center qubits have degree 3."""
    graph = _ladder_8q_target().connectivity_graph()
    corners = [0, 3, 4, 7]
    inner = [1, 2, 5, 6]
    for q in corners:
        assert len(graph[q]) == 2
    for q in inner:
        assert len(graph[q]) == 3


@gpu
@requires_gpu
def test_8qubit_single_qubit_excitation():
    """A pi pulse on q0 excites it within the full 256-dimensional register."""
    target = _ladder_8q_target()

    # The simulated register is sized by the qudits the kernel *allocates*
    # (its 8 arguments), not by the ones it drives. So q0 alone gets a pi
    # pulse while q1..q7 idle in |0>, and the full 8-qubit (256-dim) register
    # still evolves.
    @pulse.kernel
    def excite_q0(q0, q1, q2, q3, q4, q5, q6, q7):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0), t0)

    refs = [pulse.qudit_ref() for _ in range(8)]
    result = pulse.evolve(excite_q0(*refs),
                          target=target,
                          t_start=0.0,
                          t_end=20.0,
                          num_steps=200,
                          integrator="rk4")

    state = result.final_state
    assert state.shape == (256,)
    assert np.vdot(state, state).real == pytest.approx(1.0, abs=1.0e-5)

    # A single pi pulse drives exactly one qubit, so population leaves the
    # all-ground state and lands in the single-excitation manifold (basis
    # states whose index is a power of two). This holds regardless of the
    # MSB/LSB ordering convention.
    probs = np.abs(state)**2
    single_excitation = float(sum(probs[1 << b] for b in range(8)))
    assert probs[0] < 0.1  # left the all-ground state
    assert single_excitation > 0.5  # one qubit excited
