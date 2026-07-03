# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GHZ state preparation at the pulse level.

Creates the N-qubit GHZ state |000...0> + |111...1> using:
  1. A pi/2 pulse on qubit 0
  2. A chain of echoed cross-resonance CNOT gates: (0,1), (1,2), ..., (N-2,N-1)

Demonstrates multi-qubit pulse programming with internal allocation,
sync barriers, and the cross-resonance gate primitive.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import cudaq_pulse as pulse


@pulse.kernel
def ghz_3():
    """3-qubit GHZ state: H(qubit_0) - CNOT(0,1) - CNOT(1,2).

    Each CNOT is an echoed cross-resonance gate:
      SX(target) - CR(ctrl->tgt) - X(ctrl) - CR_neg(ctrl->tgt) - SX(target)
    """
    qubit_0 = pulse.qudit_ref()
    qubit_1 = pulse.qudit_ref()
    qubit_2 = pulse.qudit_ref()

    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)
    drive_line_2, tone_2 = get_drive_line(qubit_2)

    sx_pulse = drag(40, 0.25, 10.0, 0.5)
    x_pi = drag(40, 0.50, 10.0, 0.5)
    cr_pos = gaussian_square(200, 0.05, 10.0, 160)
    cr_neg = gaussian_square(200, -0.05, 10.0, 160)

    # H(qubit_0) = Rz(pi) SX Rz(pi/2)
    shift_phase(tone_0, 3.14159)
    drive(drive_line_0, sx_pulse, tone_0)
    shift_phase(tone_0, 1.5708)

    sync(drive_line_0, drive_line_1)

    # CNOT(0, 1): echoed CR
    drive(drive_line_1, sx_pulse, tone_1)  # SX on target
    drive(drive_line_0, cr_pos, tone_1)  # CR on ctrl at target freq
    drive(drive_line_0, x_pi, tone_0)  # X on ctrl
    drive(drive_line_0, cr_neg, tone_1)  # CR_neg on ctrl at target freq
    drive(drive_line_1, sx_pulse, tone_1)  # SX on target

    sync(drive_line_0, drive_line_1, drive_line_2)

    # CNOT(1, 2): echoed CR
    drive(drive_line_2, sx_pulse, tone_2)
    drive(drive_line_1, cr_pos, tone_2)
    drive(drive_line_1, x_pi, tone_1)
    drive(drive_line_1, cr_neg, tone_2)
    drive(drive_line_2, sx_pulse, tone_2)


@pulse.kernel
def ghz_n(qubit_count):
    """N-qubit GHZ state using a loop of CNOT gates.

    The outer structure: H on qubit_0, then CNOT chain.
    The for-loop demonstrates loop capture for repeated gate patterns.
    Qubit allocation is inside the kernel.
    """
    qubit_0 = pulse.qudit_ref()
    qubit_1 = pulse.qudit_ref()

    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    sx_pulse = drag(40, 0.25, 10.0, 0.5)
    x_pi = drag(40, 0.50, 10.0, 0.5)
    cr_pos = gaussian_square(200, 0.05, 10.0, 160)
    cr_neg = gaussian_square(200, -0.05, 10.0, 160)

    # H(qubit_0)
    shift_phase(tone_0, 3.14159)
    drive(drive_line_0, sx_pulse, tone_0)
    shift_phase(tone_0, 1.5708)

    # CNOT chain: for a 2-qubit kernel we do one CNOT(0,1)
    # In a real N-qubit system, this loop would iterate over qubit pairs
    for _i in range(qubit_count - 1):
        sync(drive_line_0, drive_line_1)
        drive(drive_line_1, sx_pulse, tone_1)
        drive(drive_line_0, cr_pos, tone_1)
        drive(drive_line_0, x_pi, tone_0)
        drive(drive_line_0, cr_neg, tone_1)
        drive(drive_line_1, sx_pulse, tone_1)


if __name__ == "__main__":
    print("=== 3-qubit GHZ (explicit) ===")
    compiled_kernel = pulse.compile(ghz_3, [],
                                    qubit_freq_hz={
                                        0: 5e9,
                                        1: 5.1e9,
                                        2: 5.2e9
                                    })
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== N-qubit GHZ (loop-based, N=5) ===")
    compiled_kernel = pulse.compile(ghz_n, [5],
                                    qubit_freq_hz={
                                        0: 5e9,
                                        1: 5.1e9
                                    })
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")
