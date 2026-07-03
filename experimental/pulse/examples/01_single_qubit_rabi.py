# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-qubit Rabi oscillation.

Drives a qubit with a square pulse of varying amplitude to trace out
Rabi oscillations. The simplest end-to-end example, showing both
external and internal qudit allocation styles.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import cudaq_pulse as pulse


@pulse.kernel
def rabi_external(qubit, duration, amplitude):
    """Rabi with qudit allocated externally and passed in."""
    drive_line, tone = get_drive_line(qubit)
    drive(drive_line, square(duration, amplitude), tone)


@pulse.kernel
def rabi_internal(duration, amplitude):
    """Rabi with qudit allocated inside the kernel."""
    qubit = pulse.qudit_ref()
    drive_line, tone = get_drive_line(qubit)
    drive(drive_line, square(duration, amplitude), tone)


@pulse.kernel
def rabi_with_readout(duration, amplitude):
    """Rabi experiment with measurement — the full circuit."""
    qubit = pulse.qudit_ref()
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    drive(drive_line, square(duration, amplitude), drive_tone)

    sync(drive_line, readout_line)
    readout(readout_line, square(1000, 0.05), readout_tone)


if __name__ == "__main__":
    print("=== External allocation ===")
    compiled_kernel = pulse.compile(rabi_external,
                                    [pulse.qudit_ref(), 100, 0.5],
                                    qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== Internal allocation ===")
    compiled_kernel = pulse.compile(rabi_internal, [100, 0.5],
                                    qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)

    print("\n=== Amplitude sweep (Rabi oscillation) ===")
    for amplitude in [0.1, 0.2, 0.3, 0.4, 0.5]:
        compiled_kernel = pulse.compile(rabi_internal, [100, amplitude],
                                        qubit_freq_hz={0: 5e9})
        print(f"  amplitude={amplitude:.1f}: compiled in "
              f"{compiled_kernel.metrics.total_ms:.3f} ms")

    print("\n=== With readout ===")
    compiled_kernel = pulse.compile(rabi_with_readout, [100, 0.5],
                                    qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")
