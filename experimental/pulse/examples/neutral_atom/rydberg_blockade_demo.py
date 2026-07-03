#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-atom Rydberg blockade demonstration.

Demonstrates:
  - Two atoms within the blockade radius
  - The Rydberg interaction prevents double excitation
  - Comparison of interaction strength at different spacings

When atoms are within the blockade radius (R_b), the interaction energy
V = C6/r^6 >> Omega, preventing both atoms from being excited to |r>.
This is the basis for Rydberg quantum gates.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import math

import cudaq_pulse as pulse
from cudaq_pulse.targets.rydberg import rydberg_chain, RydbergAtom, RydbergTarget

RABI_MHZ = 4.0

print("=== Rydberg Blockade Physics ===\n")

# Compute blockade radius
reference_chain = rydberg_chain(2, spacing_um=6.0, global_rabi_mhz=RABI_MHZ)
R_b = reference_chain.blockade_radius()
print(f"Blockade radius R_b = (C6/Omega)^(1/6) = {R_b:.2f} um")
print(f"  C6 = {reference_chain.c6:.0f} * 2pi MHz um^6 (Rb-87, 70S_1/2)")
print(f"  Omega = {RABI_MHZ} MHz")

print("\n--- Interaction strength vs spacing ---")
for spacing in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0]:
    chain = rydberg_chain(2, spacing_um=spacing, global_rabi_mhz=RABI_MHZ)
    V = chain.interaction_strength(chain.atoms[0], chain.atoms[1])
    ratio = V / RABI_MHZ
    regime = "BLOCKADE" if spacing < R_b else "weak"
    print(
        f"  r = {spacing:5.1f} um  |  V = {V:10.2f} MHz  |  V/Omega = {ratio:8.1f}  |  {regime}"
    )

# Build a pulse program for blockade demo
print("\n--- Two-atom blockade pulse program ---")

close_chain = rydberg_chain(2, spacing_um=4.0, global_rabi_mhz=RABI_MHZ)
V_close = close_chain.interaction_strength(close_chain.atoms[0],
                                           close_chain.atoms[1])
target = close_chain.to_target()

print(f"\nSpacing = 4.0 um (within blockade radius)")
print(f"V = {V_close:.1f} MHz >> Omega = {RABI_MHZ} MHz")
print(f"Blockade regime: double excitation |rr> strongly suppressed")

pi_duration = int(1000.0 / (2.0 * RABI_MHZ))
pi_amplitude = complex(RABI_MHZ / 10.0, 0)


@pulse.kernel
def blockade_2atom(qubit_0, qubit_1):
    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    # Global pi pulse: creates |W> = (|gr> + |rg>)/sqrt(2) in blockade regime
    pi_pulse = square(pi_duration, pi_amplitude)
    drive(drive_line_0, pi_pulse, tone_0)
    drive(drive_line_1, pi_pulse, tone_1)

    sync(drive_line_0, drive_line_1)

    wait(drive_line_0, 100)
    wait(drive_line_1, 100)


compiled_kernel = pulse.compile(
    blockade_2atom,
    [pulse.qudit_ref(), pulse.qudit_ref()],
    clock_ghz=1.0,
    qubit_freq_hz={
        0: RABI_MHZ * 1e6,
        1: RABI_MHZ * 1e6
    },
)

print(compiled_kernel.mlir)

metrics = compiled_kernel.metrics
print(f"\n=== Compile metrics ===")
print(f"  Trace   : {metrics.trace_ms:.3f} ms")
print(f"  Passes  : {metrics.passes_ms:.3f} ms")
print(f"  Schedule: {metrics.schedule_ms:.3f} ms")
print(f"  Total   : {metrics.total_ms:.3f} ms")
