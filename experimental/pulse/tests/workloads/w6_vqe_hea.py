# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W6: VQE hardware-efficient ansatz, single layer, 4 qubits.

Alternating Ry rotations (per-qubit) and linear-chain CNOTs.  The
Ry rotations use different parameters per qubit to produce a realistic
schedule (not degenerate across qubits).
"""

import math
import cudaq_pulse as pulse

NAME = "vqe_hea"
NUM_QUBITS = 4


def _cr_cnot(d_ctrl, d_tgt, t_ctrl, t_tgt):
    sx = drag(40, 0.025, 10.0, 0.5)
    cr = gaussian(200, 0.10, 50.0)
    cr_neg = gaussian(200, -0.10, 50.0)
    x_c = square(40, [0.047, 0.0])

    drive(d_tgt, sx, t_tgt)
    drive(d_ctrl, cr, t_tgt)
    drive(d_ctrl, x_c, t_ctrl)
    drive(d_ctrl, cr_neg, t_tgt)
    drive(d_tgt, sx, t_tgt)


@pulse.kernel
def build(q0, q1, q2, q3):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    d3, t3 = get_drive_line(q3)

    lines = [d0, d1, d2, d3]
    tones = [t0, t1, t2, t3]
    thetas = [0.3, 0.5, 0.7, 0.9]

    # Ry(theta_q) = Rz(-pi/2) · SX · Rz(theta) · SX · Rz(pi/2)
    sx = drag(40, 0.25, 10.0, 0.5)
    for q in range(4):
        shift_phase(tones[q], -math.pi / 2)
        drive(lines[q], sx, tones[q])
        shift_phase(tones[q], thetas[q])
        drive(lines[q], sx, tones[q])
        shift_phase(tones[q], math.pi / 2)

    # Linear-chain CNOTs: 0->1, 1->2, 2->3
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        sync(lines[a], lines[b])
        _cr_cnot(lines[a], lines[b], tones[a], tones[b])
