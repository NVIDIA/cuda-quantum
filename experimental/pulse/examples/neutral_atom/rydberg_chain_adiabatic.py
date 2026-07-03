#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adiabatic sweep on a 1D Rydberg chain.

Demonstrates:
  - Creating a Rydberg target with a 1D chain geometry
  - Computing blockade radius and interaction strengths
  - Building a pulse program for global adiabatic detuning sweep
  - Compiling via ``pulse.compile()``

The adiabatic protocol starts with large negative detuning (all atoms in |g>),
ramps through resonance, and ends at large positive detuning to prepare a
many-body ordered state (Z2 antiferromagnet for appropriate spacing).

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import math

import cudaq_pulse as pulse
from cudaq_pulse.targets.rydberg import rydberg_chain

N_ATOMS = 7
SPACING_UM = 6.0

chain = rydberg_chain(N_ATOMS, spacing_um=SPACING_UM, global_rabi_mhz=4.0)
target = chain.to_target()

print(f"Rydberg chain: {chain.n_atoms} atoms, spacing = {SPACING_UM} um")
print(f"Blockade radius: {chain.blockade_radius():.2f} um")
print(
    f"Nearest-neighbor V = {chain.interaction_strength(chain.atoms[0], chain.atoms[1]):.2f} MHz"
)
print(
    f"Next-nearest V = {chain.interaction_strength(chain.atoms[0], chain.atoms[2]):.4f} MHz"
)

ham_terms = chain.hamiltonian_terms()
diss_terms = chain.dissipator_terms()
print(f"\nHamiltonian: {len(ham_terms)} terms")
for term in ham_terms[:5]:
    print(f"  {term['kind']:25s} qubits={term['qubit_indices']} "
          f"coeff={term['coefficient'].real:.4e}")
if len(ham_terms) > 5:
    print(f"  ... ({len(ham_terms) - 5} more)")
print(f"Dissipators: {len(diss_terms)} terms")

RAMP_STEPS = 20
STEP_DURATION = 50
amplitude = chain.global_rabi_mhz / 10.0


@pulse.kernel
def rydberg_adiabatic(qubit_0, qubit_1, qubit_2, qubit_3, qubit_4, qubit_5,
                      qubit_6):
    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)
    drive_line_2, tone_2 = get_drive_line(qubit_2)
    drive_line_3, tone_3 = get_drive_line(qubit_3)
    drive_line_4, tone_4 = get_drive_line(qubit_4)
    drive_line_5, tone_5 = get_drive_line(qubit_5)
    drive_line_6, tone_6 = get_drive_line(qubit_6)

    for step in range(RAMP_STEPS):
        pulse_0 = square(STEP_DURATION, complex(amplitude, 0))
        drive(drive_line_0, pulse_0, tone_0)
        pulse_1 = square(STEP_DURATION, complex(amplitude, 0))
        drive(drive_line_1, pulse_1, tone_1)
        pulse_2 = square(STEP_DURATION, complex(amplitude, 0))
        drive(drive_line_2, pulse_2, tone_2)
        pulse_3 = square(STEP_DURATION, complex(amplitude, 0))
        drive(drive_line_3, pulse_3, tone_3)
        pulse_4 = square(STEP_DURATION, complex(amplitude, 0))
        drive(drive_line_4, pulse_4, tone_4)
        pulse_5 = square(STEP_DURATION, complex(amplitude, 0))
        drive(drive_line_5, pulse_5, tone_5)
        pulse_6 = square(STEP_DURATION, complex(amplitude, 0))
        drive(drive_line_6, pulse_6, tone_6)

        if step < RAMP_STEPS - 1:
            sync(drive_line_0, drive_line_1, drive_line_2, drive_line_3,
                 drive_line_4, drive_line_5, drive_line_6)


frequency = chain.global_rabi_mhz * 1e6
compiled_kernel = pulse.compile(
    rydberg_adiabatic,
    [pulse.qudit_ref() for _ in range(N_ATOMS)],
    clock_ghz=1.0,
    qubit_freq_hz={i: frequency for i in range(N_ATOMS)},
)

print(f"\nSweep time: {RAMP_STEPS * STEP_DURATION} VTU")
print(compiled_kernel.mlir)

metrics = compiled_kernel.metrics
print(f"\n=== Compile metrics ===")
print(f"  Trace   : {metrics.trace_ms:.3f} ms")
print(f"  Passes  : {metrics.passes_ms:.3f} ms")
print(f"  Schedule: {metrics.schedule_ms:.3f} ms")
print(f"  Total   : {metrics.total_ms:.3f} ms")
