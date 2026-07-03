# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamical decoupling pulse sequences: XY4, Uhrig, and CPMG.

Each sequence uses a different refocusing pattern to suppress different
noise spectra. All three use for-loops that capture as rolled scf.for ops
in the IR — demonstrating the kernel's loop-capture capability.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import math

import cudaq_pulse as pulse


@pulse.kernel
def xy4(sigma, amplitude_pi, tau, n_cycles):
    """XY4 dynamical decoupling: [X - tau - Y - tau - X - tau - Y - tau]^n.

    Suppresses both dephasing and amplitude noise by alternating X and Y
    refocusing axes.
    """
    qubit = pulse.qudit_ref()
    drive_line, tone = get_drive_line(qubit)

    dur_pi = int(4 * sigma)
    pi_x = gaussian(dur_pi, amplitude_pi, sigma)

    for _cycle in range(n_cycles):
        # X refocus
        drive(drive_line, pi_x, tone)
        wait(drive_line, tau)

        # Y refocus = phase shift pi/2, then pi_X, then undo phase
        shift_phase(tone, math.pi / 2)
        drive(drive_line, pi_x, tone)
        shift_phase(tone, -math.pi / 2)
        wait(drive_line, tau)

        # X refocus
        drive(drive_line, pi_x, tone)
        wait(drive_line, tau)

        # Y refocus
        shift_phase(tone, math.pi / 2)
        drive(drive_line, pi_x, tone)
        shift_phase(tone, -math.pi / 2)
        wait(drive_line, tau)


@pulse.kernel
def uhrig_dd(sigma, amplitude_pi, total_time, n_pulses):
    """Uhrig dynamical decoupling (UDD).

    Places n refocusing pulses at non-uniform intervals:
        t_j = T * sin^2(pi * j / (2n + 2))

    Optimal for pure dephasing from a soft-cutoff noise spectrum.
    The intervals are computed at compile time from n_pulses.
    """
    qubit = pulse.qudit_ref()
    drive_line, tone = get_drive_line(qubit)

    dur_pi = int(4 * sigma)
    pi_pulse = gaussian(dur_pi, amplitude_pi, sigma)

    prev_time = 0
    for j in range(n_pulses):
        frac = math.sin(math.pi * (j + 1) / (2 * n_pulses + 2))**2
        tj = int(total_time * frac)
        gap = tj - prev_time - dur_pi
        if gap > 0:
            wait(drive_line, gap)
        drive(drive_line, pi_pulse, tone)
        prev_time = tj

    remaining = total_time - prev_time
    if remaining > 0:
        wait(drive_line, remaining)


@pulse.kernel
def cpmg(sigma, amplitude_half_pi, amplitude_pi, tau, n_refocus):
    """Carr-Purcell-Meiboom-Gill: pi/2_X - [tau - pi_Y - tau]^n - pi/2_X.

    The standard multi-pulse echo for dephasing suppression.
    """
    qubit = pulse.qudit_ref()
    drive_line, tone = get_drive_line(qubit)

    dur = int(4 * sigma)
    half_pi_pulse = gaussian(dur, amplitude_half_pi, sigma)
    pi_pulse = gaussian(dur, amplitude_pi, sigma)

    drive(drive_line, half_pi_pulse, tone)

    for _i in range(n_refocus):
        wait(drive_line, tau)
        shift_phase(tone, math.pi / 2)
        drive(drive_line, pi_pulse, tone)
        shift_phase(tone, -math.pi / 2)
        wait(drive_line, tau)

    drive(drive_line, half_pi_pulse, tone)


@pulse.kernel
def knill_dd(sigma, amplitude_pi, tau):
    """Knill dynamical decoupling (KDD): a composite-pulse sequence.

    Uses 5 pi-pulses with carefully chosen phases to suppress both
    systematic over/under-rotation and dephasing.
    Phases: [pi/6, 0, pi/2, 0, pi/6]
    """
    qubit = pulse.qudit_ref()
    drive_line, tone = get_drive_line(qubit)

    dur_pi = int(4 * sigma)
    pi_pulse = gaussian(dur_pi, amplitude_pi, sigma)

    phases = [math.pi / 6, 0.0, math.pi / 2, 0.0, math.pi / 6]

    for idx in range(5):
        wait(drive_line, tau)
        shift_phase(tone, phases[idx])
        drive(drive_line, pi_pulse, tone)
        shift_phase(tone, -phases[idx])


if __name__ == "__main__":
    sigma = 10.0
    amplitude_pi = 0.50
    amplitude_half_pi = 0.25
    tau = 200

    print("=== XY4 (4 cycles) ===")
    compiled_kernel = pulse.compile(xy4, [sigma, amplitude_pi, tau, 4],
                                    qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== Uhrig DD (8 pulses, T=5000 VTU) ===")
    compiled_kernel = pulse.compile(uhrig_dd, [sigma, amplitude_pi, 5000, 8],
                                    qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)

    print("\n=== CPMG (8 refocusing pulses) ===")
    compiled_kernel = pulse.compile(
        cpmg, [sigma, amplitude_half_pi, amplitude_pi, tau, 8],
        qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)

    print("\n=== KDD (Knill DD) ===")
    compiled_kernel = pulse.compile(knill_dd, [sigma, amplitude_pi, tau],
                                    qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")
