# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""W3: 4-qubit QAOA single layer on a ring topology.

Tests schedule density: 4 parallel Rz-diagonal rotations and 4 CR
gates on a ring, requiring explicit synchronization points.
"""

import math
import cudaq_pulse as pulse

NAME = "qaoa4"
NUM_QUBITS = 4


def _cr_cnot(d_ctrl, d_tgt, t_ctrl, t_tgt):
    """Inline a CR CNOT; same op sequence as w2_cnot_cr."""
    sx = drag(40, 0.025, 10.0, 0.5)
    cr = gaussian(200, 0.10, 50.0)
    cr_neg = gaussian(200, -0.10, 50.0)
    x_c = square(40, [0.047, 0.0])

    drive(d_tgt, sx, t_tgt)
    drive(d_ctrl, cr, t_tgt)
    drive(d_ctrl, x_c, t_ctrl)
    drive(d_ctrl, cr_neg, t_tgt)
    drive(d_tgt, sx, t_tgt)


def _cost_edge(d_ctrl, d_tgt, t_ctrl, t_tgt, gamma):
    sync(d_ctrl, d_tgt)
    _cr_cnot(d_ctrl, d_tgt, t_ctrl, t_tgt)
    shift_phase(t_tgt, 2 * gamma)
    _cr_cnot(d_ctrl, d_tgt, t_ctrl, t_tgt)


@pulse.kernel
def build(q0, q1, q2, q3):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    d3, t3 = get_drive_line(q3)

    lines = [d0, d1, d2, d3]
    tones = [t0, t1, t2, t3]

    gamma = 0.3
    beta = 0.7

    # Cost layer: ZZ rotation via CNOT-Rz(2*gamma)-CNOT on ring edges.
    # Pulse kernels currently support static range loops; spell out this
    # irregular edge list so the workload stays within that contract.
    _cost_edge(d0, d1, t0, t1, gamma)
    _cost_edge(d1, d2, t1, t2, gamma)
    _cost_edge(d2, d3, t2, t3, gamma)
    _cost_edge(d3, d0, t3, t0, gamma)

    # Mixer layer: Rx(2*beta) on every qubit via SX-Rz-SX
    sx = drag(40, 0.25, 10.0, 0.5)
    for q in range(4):
        shift_phase(tones[q], math.pi / 2)
        drive(lines[q], sx, tones[q])
        shift_phase(tones[q], 2 * beta)
        drive(lines[q], sx, tones[q])
        shift_phase(tones[q], math.pi / 2)
