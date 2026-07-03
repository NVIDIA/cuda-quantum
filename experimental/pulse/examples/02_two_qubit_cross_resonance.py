# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-qubit CNOT via echoed cross-resonance.

Demonstrates cross-resonance driving: the control qubit's drive line
is modulated at the target qubit's frequency (via the target's tone).
Shows both external and internal qudit allocation.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import cudaq_pulse as pulse


@pulse.kernel
def echoed_cr_external(qubit_ctrl, qubit_tgt):
    """Echoed cross-resonance CNOT with externally allocated qudits."""
    drive_line_ctrl, tone_ctrl = get_drive_line(qubit_ctrl)
    drive_line_tgt, tone_tgt = get_drive_line(qubit_tgt)

    sync(drive_line_ctrl, drive_line_tgt)

    sx_pulse = drag(40, 0.025, 10.0, 0.5)
    x_ctrl = drag(40, 0.047, 10.0, 0.5)
    cr = gaussian_square(200, 0.10, 10.0, 160)
    cr_neg = gaussian_square(200, -0.10, 10.0, 160)

    drive(drive_line_tgt, sx_pulse, tone_tgt)
    drive(drive_line_ctrl, cr, tone_tgt)
    drive(drive_line_ctrl, x_ctrl, tone_ctrl)
    drive(drive_line_ctrl, cr_neg, tone_tgt)
    drive(drive_line_tgt, sx_pulse, tone_tgt)


@pulse.kernel
def echoed_cr_internal():
    """Echoed cross-resonance CNOT with internally allocated qudits."""
    qubit_ctrl = pulse.qudit_ref()
    qubit_tgt = pulse.qudit_ref()

    drive_line_ctrl, tone_ctrl = get_drive_line(qubit_ctrl)
    drive_line_tgt, tone_tgt = get_drive_line(qubit_tgt)

    sync(drive_line_ctrl, drive_line_tgt)

    sx_pulse = drag(40, 0.025, 10.0, 0.5)
    x_ctrl = drag(40, 0.047, 10.0, 0.5)
    cr = gaussian_square(200, 0.10, 10.0, 160)
    cr_neg = gaussian_square(200, -0.10, 10.0, 160)

    drive(drive_line_tgt, sx_pulse, tone_tgt)
    drive(drive_line_ctrl, cr, tone_tgt)
    drive(drive_line_ctrl, x_ctrl, tone_ctrl)
    drive(drive_line_ctrl, cr_neg, tone_tgt)
    drive(drive_line_tgt, sx_pulse, tone_tgt)


if __name__ == "__main__":
    print("=== External allocation ===")
    compiled_kernel = pulse.compile(
        echoed_cr_external,
        [pulse.qudit_ref(), pulse.qudit_ref()],
        qubit_freq_hz={
            0: 5e9,
            1: 5.1e9
        })
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== Internal allocation ===")
    compiled_kernel = pulse.compile(echoed_cr_internal, [],
                                    qubit_freq_hz={
                                        0: 5e9,
                                        1: 5.1e9
                                    })
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")
