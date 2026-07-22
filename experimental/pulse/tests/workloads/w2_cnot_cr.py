# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W2: CNOT via echo cross-resonance, standalone schedule.

The hero program for Figure 1.  A single schedule that can be executed
on any CR-compatible qubit pair by supplying the pair-specific tones
at call time -- no IR-level recompilation required.
"""

import cudaq_pulse as pulse

NAME = "cnot_cr"
NUM_QUBITS = 2


@pulse.kernel
def build(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)

    sync(d0, d1)

    sx = drag(40, 0.025, 10.0, 0.5)
    cr = gaussian(200, 0.10, 50.0)
    cr_neg = gaussian(200, -0.10, 50.0)
    x_ctrl = square(40, [0.047, 0.0])

    # Step 1: SX on target at target's tone
    drive(d1, sx, t1)
    # Step 2: CR drive -- control's line at target's tone
    drive(d0, cr, t1)
    # Step 3: X echo on control at control's tone
    drive(d0, x_ctrl, t0)
    # Step 4: negative-amplitude CR drive
    drive(d0, cr_neg, t1)
    # Step 5: SX on target closes the echo
    drive(d1, sx, t1)
