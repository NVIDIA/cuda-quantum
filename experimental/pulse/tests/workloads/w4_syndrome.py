# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W4: Surface-code X-syndrome measurement cycle.

One ancilla + four data qubits, sequential CNOT ladder, mid-circuit
measurement of ancilla.  Tests pulse.sync across 5 lines, explicit
readout, and the stability of schedule metrics under sync resolution.
"""

import math
import cudaq_pulse as pulse

NAME = "syndrome"
NUM_QUBITS = 5


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
def build(q_anc, q_d0, q_d1, q_d2, q_d3):
    # Ancilla = q_anc, data = q_d0..q_d3
    d_anc, t_anc = get_drive_line(q_anc)
    d0, t0 = get_drive_line(q_d0)
    d1, t1 = get_drive_line(q_d1)
    d2, t2 = get_drive_line(q_d2)
    d3, t3 = get_drive_line(q_d3)
    r_anc, rt_anc = get_readout_line(q_anc)

    data_lines = [d0, d1, d2, d3]
    data_tones = [t0, t1, t2, t3]

    # Initialise ancilla in |+>: Hadamard via Rz(pi/2)-SX-Rz(pi/2)
    sx = drag(40, 0.25, 10.0, 0.5)
    shift_phase(t_anc, math.pi / 2)
    drive(d_anc, sx, t_anc)
    shift_phase(t_anc, math.pi / 2)

    # CNOT ladder: ancilla -> each data qubit
    for i in range(4):
        sync(d_anc, data_lines[i])
        _cr_cnot(d_anc, data_lines[i], t_anc, data_tones[i])

    # Close with H on ancilla
    shift_phase(t_anc, math.pi / 2)
    drive(d_anc, sx, t_anc)
    shift_phase(t_anc, math.pi / 2)

    # Sync ancilla drive line with readout line before measuring
    sync(d_anc, r_anc)

    # Mid-circuit measurement of ancilla
    ro = square(1000, [0.05, 0.0])
    readout(r_anc, ro, rt_anc, "iq")
