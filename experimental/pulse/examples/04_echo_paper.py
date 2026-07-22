# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical spin-echo example.

Implements the Hahn echo sequence: pi/2 - tau - pi - tau - pi/2,
which refocuses dephasing due to static frequency detuning and
low-frequency noise.

This is the canonical echo example from the cudaq-pulse design plan.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import numpy as np

import cudaq_pulse as pulse


@pulse.kernel
def hahn_echo(sigma, amp_half_pi, amp_pi, tau):
    """Hahn echo: pi/2_X - tau - pi_X - tau - pi/2_X."""
    qubit = pulse.qudit_ref()
    drive_line, tone = get_drive_line(qubit)

    dur_half_pi = int(4 * sigma)
    dur_pi = int(4 * sigma)

    half_pi_envelope = gaussian(dur_half_pi, amp_half_pi, sigma)
    pi_envelope = gaussian(dur_pi, amp_pi, sigma)

    # pi/2 pulse about X
    drive(drive_line, half_pi_envelope, tone)

    # Free evolution
    wait(drive_line, tau)

    # pi refocusing pulse about X
    drive(drive_line, pi_envelope, tone)

    # Free evolution
    wait(drive_line, tau)

    # Final pi/2 pulse about X
    drive(drive_line, half_pi_envelope, tone)


@pulse.kernel
def cpmg_echo(sigma, amp_half_pi, amp_pi, tau, n_refocus):
    """CPMG dynamical decoupling: pi/2_X - [tau - pi_Y - tau]^n - pi/2_X."""
    qubit = pulse.qudit_ref()
    drive_line, tone = get_drive_line(qubit)

    dur_half_pi = int(4 * sigma)
    dur_pi = int(4 * sigma)

    half_pi_envelope = gaussian(dur_half_pi, amp_half_pi, sigma)
    pi_envelope = gaussian(dur_pi, amp_pi, sigma)

    # Initial pi/2 about X
    drive(drive_line, half_pi_envelope, tone)

    for _i in range(n_refocus):
        wait(drive_line, tau)
        # pi about Y = phase shift of pi/2 then pi_X
        shift_phase(tone, np.pi / 2)
        drive(drive_line, pi_envelope, tone)
        shift_phase(tone, -np.pi / 2)
        wait(drive_line, tau)

    # Final pi/2 about X
    drive(drive_line, half_pi_envelope, tone)


def main():
    sigma = 10.0  # Gaussian sigma in VTU
    amp_half_pi = 0.25
    amp_pi = 0.50
    tau = 500  # VTU

    print("=== Hahn Echo ===")
    compiled_kernel = pulse.compile(hahn_echo,
                                    [sigma, amp_half_pi, amp_pi, tau],
                                    qubit_freq_hz={0: 5e9})
    total_hahn = 2 * int(4 * sigma) + int(4 * sigma) + 2 * tau
    print(f"  Expected duration: {total_hahn} VTU")
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== CPMG-4 Echo ===")
    n_refocus = 4
    compiled_kernel = pulse.compile(
        cpmg_echo, [sigma, amp_half_pi, amp_pi, tau, n_refocus],
        qubit_freq_hz={0: 5e9})
    total_cpmg = 2 * int(4 * sigma) + n_refocus * (int(4 * sigma) + 2 * tau)
    print(f"  Expected duration: {total_cpmg} VTU")
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")


if __name__ == "__main__":
    main()
