# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readout, measurement, and measurement-conditioned branching.

Shows how to use readout channels, acquire IQ data, and conditionally
branch on measurement outcomes — the only non-deterministic control flow
allowed in pulse kernels.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import cudaq_pulse as pulse


@pulse.kernel
def measure_single_qubit():
    """Prepare |1> and measure.

    Demonstrates the readout channel: get_readout_line returns a
    (readout_line, tone) pair, and readout() acquires IQ data.
    """
    qubit = pulse.qudit_ref()
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    x_pulse = drag(40, 0.5, 10.0, 0.5)
    drive(drive_line, x_pulse, drive_tone)

    sync(drive_line, readout_line)

    readout_envelope = square(1000, 0.05)
    readout(readout_line, readout_envelope, readout_tone)


@pulse.kernel
def active_reset():
    """Active reset: measure, and if excited, apply an X pulse.

    This is the only context where `if` on a measurement result is
    allowed in a pulse kernel. The branch becomes an scf.if in the IR.
    """
    qubit = pulse.qudit_ref()
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    readout_envelope = square(1000, 0.05)
    result = readout(readout_line, readout_envelope, readout_tone)

    sync(drive_line, readout_line)

    if result:
        x_pulse = drag(40, 0.5, 10.0, 0.5)
        drive(drive_line, x_pulse, drive_tone)


@pulse.kernel
def repeated_measurement(n_shots):
    """Repeated preparation and measurement in a single kernel.

    A for loop wrapping prepare-measure cycles.
    """
    qubit = pulse.qudit_ref()
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    x_half = drag(40, 0.25, 10.0, 0.5)

    for _shot in range(n_shots):
        drive(drive_line, x_half, drive_tone)
        sync(drive_line, readout_line)

        readout(readout_line, square(1000, 0.05), readout_tone)

        sync(drive_line, readout_line)
        wait(drive_line, 500)


@pulse.kernel
def ramsey_with_readout(tau):
    """Ramsey experiment: pi/2 - tau - pi/2 - measure.

    Demonstrates the full single-qubit characterization loop with readout.
    """
    qubit = pulse.qudit_ref()
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    half_pi = drag(40, 0.25, 10.0, 0.5)

    drive(drive_line, half_pi, drive_tone)

    wait(drive_line, tau)

    drive(drive_line, half_pi, drive_tone)

    sync(drive_line, readout_line)
    readout(readout_line, square(1000, 0.05), readout_tone)


if __name__ == "__main__":
    print("=== Measure single qubit ===")
    compiled_kernel = pulse.compile(measure_single_qubit, [],
                                    qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)

    print("\n=== Active reset (measurement-conditioned branch) ===")
    compiled_kernel = pulse.compile(active_reset, [], qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)

    print("\n=== Repeated measurement (5 shots) ===")
    compiled_kernel = pulse.compile(repeated_measurement, [5],
                                    qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)

    print("\n=== Ramsey with readout (tau=200) ===")
    compiled_kernel = pulse.compile(ramsey_with_readout, [200],
                                    qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")
