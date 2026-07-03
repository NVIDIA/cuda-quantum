# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W5: Dynamical-decoupling CPMG-8 chain.

Single qubit, 8 X gates with equal waits between them.  Tests the
scheduler on a long idle-wait-idle schedule, and tests waveform reuse
(the same X waveform plays 8 times).  The for-loop is captured as
scf.for in the IR.
"""

import cudaq_pulse as pulse

NAME = "dd_cpmg8"
NUM_QUBITS = 1


@pulse.kernel
def build(q0):
    d0, t0 = get_drive_line(q0)
    x = square(40, [0.047, 0.0])
    wait(d0, 100)
    for i in range(8):
        drive(d0, x, t0)
        wait(d0, 200)
