# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-qubit programs with sync barriers and qvec_ref allocation.

Shows the two allocation styles (single qudit_ref vs vectorized qvec_ref),
the sync primitive for cross-line coordination, and patterns for building
multi-qubit pulse programs.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import cudaq_pulse as pulse

# ── 1. Explicit multi-qudit allocation ───────────────────────────────────────


@pulse.kernel
def two_qubit_simultaneous():
    """Drive two qubits simultaneously, then sync them.

    Each qudit is allocated individually inside the kernel.
    """
    qubit_0 = pulse.qudit_ref()
    qubit_1 = pulse.qudit_ref()

    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    drive(drive_line_0, gaussian(40, 0.25, 10.0), tone_0)
    drive(drive_line_1, gaussian(40, 0.30, 10.0), tone_1)

    sync(drive_line_0, drive_line_1)

    drive(drive_line_0, gaussian(40, 0.25, 10.0), tone_0)
    drive(drive_line_1, gaussian(40, 0.30, 10.0), tone_1)


# ── 2. External allocation — qudits passed as arguments ──────────────────────


@pulse.kernel
def parametric_pi_half(qubit_0, qubit_1, amplitude_0, amplitude_1):
    """Calibration-friendly: pass external qudit refs and amplitudes in."""
    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    drive(drive_line_0, drag(40, amplitude_0, 10.0, 0.5), tone_0)
    drive(drive_line_1, drag(40, amplitude_1, 10.0, 0.5), tone_1)


# ── 3. qvec_ref — vectorized qudit allocation ────────────────────────────────


@pulse.kernel
def qvec_simultaneous_drive(qubit_count, amplitude):
    """Drive N qubits in parallel using a qvec_ref.

    pulse.qvec_ref(qubit_count) allocates a contiguous vector; individual
    qudits are accessed by indexing.
    """
    qubit_vector = pulse.qvec_ref(qubit_count)
    drive_line_0, tone_0 = get_drive_line(qubit_vector[0])
    drive_line_1, tone_1 = get_drive_line(qubit_vector[1])

    drive(drive_line_0, gaussian(40, amplitude, 10.0), tone_0)
    drive(drive_line_1, gaussian(40, amplitude, 10.0), tone_1)
    sync(drive_line_0, drive_line_1)


# ── 4. Sync patterns ─────────────────────────────────────────────────────────


@pulse.kernel
def sync_after_asymmetric_ops():
    """Sync after different-duration ops to re-align timelines.

    Without sync, the two lines would diverge in time.
    """
    qubit_0 = pulse.qudit_ref()
    qubit_1 = pulse.qudit_ref()

    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    drive(drive_line_0, square(200, 0.1), tone_0)
    drive(drive_line_1, gaussian(40, 0.25, 10.0), tone_1)

    sync(drive_line_0, drive_line_1)

    drive(drive_line_0, gaussian(40, 0.25, 10.0), tone_0)
    drive(drive_line_1, gaussian(40, 0.25, 10.0), tone_1)


@pulse.kernel
def staggered_readout():
    """Stagger operations across qubits with wait + sync.

    Drive qubit_0, wait, then sync with qubit_1 before a joint operation.
    """
    qubit_0 = pulse.qudit_ref()
    qubit_1 = pulse.qudit_ref()

    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    drive(drive_line_0, gaussian(40, 0.5, 10.0), tone_0)
    wait(drive_line_0, 500)

    drive(drive_line_1, gaussian(40, 0.5, 10.0), tone_1)

    sync(drive_line_0, drive_line_1)

    drive(drive_line_0, gaussian(40, 0.25, 10.0), tone_0)
    drive(drive_line_1, gaussian(40, 0.25, 10.0), tone_1)


if __name__ == "__main__":
    print("=== Two-qubit simultaneous drive (internal alloc) ===")
    compiled_kernel = pulse.compile(two_qubit_simultaneous, [],
                                    qubit_freq_hz={
                                        0: 5e9,
                                        1: 5.1e9
                                    })
    print(compiled_kernel.mlir)

    print("\n=== Parametric pi/2 (external alloc) ===")
    compiled_kernel = pulse.compile(
        parametric_pi_half,
        [pulse.qudit_ref(), pulse.qudit_ref(), 0.25, 0.28],
        qubit_freq_hz={
            0: 5e9,
            1: 5.1e9
        })
    print(compiled_kernel.mlir)

    print("\n=== qvec simultaneous drive ===")
    compiled_kernel = pulse.compile(
        qvec_simultaneous_drive, [4, 0.3],
        qubit_freq_hz={i: 5e9 + i * 0.1e9 for i in range(4)})
    print(compiled_kernel.mlir)

    print("\n=== Sync after asymmetric ops ===")
    compiled_kernel = pulse.compile(sync_after_asymmetric_ops, [],
                                    qubit_freq_hz={
                                        0: 5e9,
                                        1: 5.1e9
                                    })
    print(compiled_kernel.mlir)

    print("\n=== Staggered readout ===")
    compiled_kernel = pulse.compile(staggered_readout, [],
                                    qubit_freq_hz={
                                        0: 5e9,
                                        1: 5.1e9
                                    })
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")
