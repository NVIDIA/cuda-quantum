#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One QEC syndrome extraction cycle on the Krinner d=3 surface code.

Demonstrates:
  - Building a multi-qubit pulse program on the 17-qubit target
  - Z-stabilizer and X-stabilizer measurement rounds
  - Sync barriers between stabilizer groups
  - Full pipeline: verify -> schedule -> pulse_to_operator

The 17-qubit layout (Krinner et al.):
  Data qubits:     D1-D9 (indices 0-8)
  Z-ancillas:      Z1-Z4 (indices 9-12)
  X-ancillas:      X1-X4 (indices 13-16)

NOTE: This example uses the advanced internal API (``_to_program``) because
``run_pulse_to_operator`` operates on ``Program`` objects — a specialized
pass not available in the standard C++ compilation pipeline.
"""

import math

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program
from cudaq_pulse.passes.verify import verify
from cudaq_pulse.passes.scheduling import schedule_alap
from cudaq_pulse.passes.pulse_to_operator import run_pulse_to_operator
from cudaq_pulse.targets import transmon_krinner_17q

target = transmon_krinner_17q()

# Stabilizer definitions (ancilla -> data qubit pairs)
Z_STABILIZERS = {
    9: [0, 1],  # Z1 (weight-2)
    10: [0, 1, 3, 4],  # Z2 (weight-4)
    11: [4, 5, 7, 8],  # Z3 (weight-4)
    12: [7, 8],  # Z4 (weight-2)
}
X_STABILIZERS = {
    13: [0, 3],  # X1 (weight-2)
    14: [1, 2, 4, 5],  # X2 (weight-4)
    15: [3, 4, 6, 7],  # X3 (weight-4)
    16: [5, 8],  # X4 (weight-2)
}

print(f"Target: {target.name}")
print(
    f"Z-stabilizers: {len(Z_STABILIZERS)}, X-stabilizers: {len(X_STABILIZERS)}")

# Collect drive params for all qubits (captured as closures)
all_qubits = sorted(target.qubits.keys())
all_drive_params = {qi: target.get_drive_params(qi) for qi in all_qubits}


@pulse.kernel
def surface_code_cycle(
    qubit_0,
    qubit_1,
    qubit_2,
    qubit_3,
    qubit_4,
    qubit_5,
    qubit_6,
    qubit_7,
    qubit_8,
    qubit_9,
    qubit_10,
    qubit_11,
    qubit_12,
    qubit_13,
    qubit_14,
    qubit_15,
    qubit_16,
):
    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)
    drive_line_2, tone_2 = get_drive_line(qubit_2)
    drive_line_3, tone_3 = get_drive_line(qubit_3)
    drive_line_4, tone_4 = get_drive_line(qubit_4)
    drive_line_5, tone_5 = get_drive_line(qubit_5)
    drive_line_6, tone_6 = get_drive_line(qubit_6)
    drive_line_7, tone_7 = get_drive_line(qubit_7)
    drive_line_8, tone_8 = get_drive_line(qubit_8)
    drive_line_9, tone_9 = get_drive_line(qubit_9)
    drive_line_10, tone_10 = get_drive_line(qubit_10)
    drive_line_11, tone_11 = get_drive_line(qubit_11)
    drive_line_12, tone_12 = get_drive_line(qubit_12)
    drive_line_13, tone_13 = get_drive_line(qubit_13)
    drive_line_14, tone_14 = get_drive_line(qubit_14)
    drive_line_15, tone_15 = get_drive_line(qubit_15)
    drive_line_16, tone_16 = get_drive_line(qubit_16)

    # -- Z-stabilizer round --

    # Z1 (ancilla 9): Hadamard, CZ with data [0, 1], Hadamard
    shift_phase(tone_9, math.pi / 2)
    sx_pulse = drag(20, all_drive_params[9].get("x_amp", 0.44),
                    all_drive_params[9].get("x_sigma", 5.0),
                    all_drive_params[9].get("x_beta", 0.7))
    drive(drive_line_9, sx_pulse, tone_9)
    shift_phase(tone_9, math.pi / 2)

    sync(drive_line_9, drive_line_0)
    cr = gaussian(98, 0.32, 24.0)
    drive(drive_line_0, cr, tone_0)
    x_echo = drag(20, all_drive_params[0].get("x_amp", 0.44),
                  all_drive_params[0].get("x_sigma", 5.0),
                  all_drive_params[0].get("x_beta", 0.7))
    drive(drive_line_9, x_echo, tone_9)
    cr_neg = gaussian(98, -0.32, 24.0)
    drive(drive_line_0, cr_neg, tone_0)
    drive(drive_line_9, x_echo, tone_9)

    sync(drive_line_9, drive_line_1)
    cr = gaussian(98, 0.32, 24.0)
    drive(drive_line_1, cr, tone_1)
    x_echo = drag(20, all_drive_params[1].get("x_amp", 0.44),
                  all_drive_params[1].get("x_sigma", 5.0),
                  all_drive_params[1].get("x_beta", 0.7))
    drive(drive_line_9, x_echo, tone_9)
    cr_neg = gaussian(98, -0.32, 24.0)
    drive(drive_line_1, cr_neg, tone_1)
    drive(drive_line_9, x_echo, tone_9)

    shift_phase(tone_9, math.pi / 2)
    sx_pulse = drag(20, all_drive_params[9].get("x_amp", 0.44),
                    all_drive_params[9].get("x_sigma", 5.0),
                    all_drive_params[9].get("x_beta", 0.7))
    drive(drive_line_9, sx_pulse, tone_9)
    shift_phase(tone_9, math.pi / 2)

    # Z2 (ancilla 10): Hadamard, CZ with data [0, 1, 3, 4], Hadamard
    shift_phase(tone_10, math.pi / 2)
    sx_pulse = drag(20, all_drive_params[10].get("x_amp", 0.44),
                    all_drive_params[10].get("x_sigma", 5.0),
                    all_drive_params[10].get("x_beta", 0.7))
    drive(drive_line_10, sx_pulse, tone_10)
    shift_phase(tone_10, math.pi / 2)

    sync(drive_line_10, drive_line_0)
    cr = gaussian(98, 0.32, 24.0)
    drive(drive_line_0, cr, tone_0)
    x_echo = drag(20, all_drive_params[0].get("x_amp", 0.44),
                  all_drive_params[0].get("x_sigma", 5.0),
                  all_drive_params[0].get("x_beta", 0.7))
    drive(drive_line_10, x_echo, tone_10)
    cr_neg = gaussian(98, -0.32, 24.0)
    drive(drive_line_0, cr_neg, tone_0)
    drive(drive_line_10, x_echo, tone_10)

    sync(drive_line_10, drive_line_1)
    cr = gaussian(98, 0.32, 24.0)
    drive(drive_line_1, cr, tone_1)
    x_echo = drag(20, all_drive_params[1].get("x_amp", 0.44),
                  all_drive_params[1].get("x_sigma", 5.0),
                  all_drive_params[1].get("x_beta", 0.7))
    drive(drive_line_10, x_echo, tone_10)
    cr_neg = gaussian(98, -0.32, 24.0)
    drive(drive_line_1, cr_neg, tone_1)
    drive(drive_line_10, x_echo, tone_10)

    sync(drive_line_10, drive_line_3)
    cr = gaussian(98, 0.32, 24.0)
    drive(drive_line_3, cr, tone_3)
    x_echo = drag(20, all_drive_params[3].get("x_amp", 0.44),
                  all_drive_params[3].get("x_sigma", 5.0),
                  all_drive_params[3].get("x_beta", 0.7))
    drive(drive_line_10, x_echo, tone_10)
    cr_neg = gaussian(98, -0.32, 24.0)
    drive(drive_line_3, cr_neg, tone_3)
    drive(drive_line_10, x_echo, tone_10)

    sync(drive_line_10, drive_line_4)
    cr = gaussian(98, 0.32, 24.0)
    drive(drive_line_4, cr, tone_4)
    x_echo = drag(20, all_drive_params[4].get("x_amp", 0.44),
                  all_drive_params[4].get("x_sigma", 5.0),
                  all_drive_params[4].get("x_beta", 0.7))
    drive(drive_line_10, x_echo, tone_10)
    cr_neg = gaussian(98, -0.32, 24.0)
    drive(drive_line_4, cr_neg, tone_4)
    drive(drive_line_10, x_echo, tone_10)

    shift_phase(tone_10, math.pi / 2)
    sx_pulse = drag(20, all_drive_params[10].get("x_amp", 0.44),
                    all_drive_params[10].get("x_sigma", 5.0),
                    all_drive_params[10].get("x_beta", 0.7))
    drive(drive_line_10, sx_pulse, tone_10)
    shift_phase(tone_10, math.pi / 2)

    # Global sync before X round
    sync(drive_line_0, drive_line_1, drive_line_2, drive_line_3, drive_line_4,
         drive_line_5, drive_line_6, drive_line_7, drive_line_8, drive_line_9,
         drive_line_10, drive_line_11, drive_line_12, drive_line_13,
         drive_line_14, drive_line_15, drive_line_16)

    # -- X-stabilizer round (first two for brevity) --

    # X1 (ancilla 13): CZ with data [0, 3]
    sync(drive_line_13, drive_line_0)
    cr = gaussian(98, 0.32, 24.0)
    drive(drive_line_13, cr, tone_13)
    x_echo = drag(20, all_drive_params[0].get("x_amp", 0.44),
                  all_drive_params[0].get("x_sigma", 5.0),
                  all_drive_params[0].get("x_beta", 0.7))
    drive(drive_line_0, x_echo, tone_0)
    cr_neg = gaussian(98, -0.32, 24.0)
    drive(drive_line_13, cr_neg, tone_13)
    drive(drive_line_0, x_echo, tone_0)

    sync(drive_line_13, drive_line_3)
    cr = gaussian(98, 0.32, 24.0)
    drive(drive_line_13, cr, tone_13)
    x_echo = drag(20, all_drive_params[3].get("x_amp", 0.44),
                  all_drive_params[3].get("x_sigma", 5.0),
                  all_drive_params[3].get("x_beta", 0.7))
    drive(drive_line_3, x_echo, tone_3)
    cr_neg = gaussian(98, -0.32, 24.0)
    drive(drive_line_13, cr_neg, tone_13)
    drive(drive_line_3, x_echo, tone_3)

    # X4 (ancilla 16): CZ with data [5, 8]
    sync(drive_line_16, drive_line_5)
    cr = gaussian(98, 0.32, 24.0)
    drive(drive_line_16, cr, tone_16)
    x_echo = drag(20, all_drive_params[5].get("x_amp", 0.44),
                  all_drive_params[5].get("x_sigma", 5.0),
                  all_drive_params[5].get("x_beta", 0.7))
    drive(drive_line_5, x_echo, tone_5)
    cr_neg = gaussian(98, -0.32, 24.0)
    drive(drive_line_16, cr_neg, tone_16)
    drive(drive_line_5, x_echo, tone_5)

    sync(drive_line_16, drive_line_8)
    cr = gaussian(98, 0.32, 24.0)
    drive(drive_line_16, cr, tone_16)
    x_echo = drag(20, all_drive_params[8].get("x_amp", 0.44),
                  all_drive_params[8].get("x_sigma", 5.0),
                  all_drive_params[8].get("x_beta", 0.7))
    drive(drive_line_8, x_echo, tone_8)
    cr_neg = gaussian(98, -0.32, 24.0)
    drive(drive_line_16, cr_neg, tone_16)
    drive(drive_line_8, x_echo, tone_8)


ir = surface_code_cycle(*(pulse.qudit_ref() for _ in range(17)))
program = _to_program(
    ir,
    clock_ghz=2.0,
    qubit_freq_hz={qi: target.qubits[qi].frequency_hz for qi in all_qubits},
)

issues = verify(program)
errors = [i for i in issues if i.severity == "error"]
print(f"\nVerification: {len(issues)} total issues, {len(errors)} errors")
for issue in issues[:5]:
    print(f"  {issue}")

events, metrics = schedule_alap(program)
print(
    f"\nScheduled: {metrics.op_count} ops, "
    f"total = {metrics.total_length_ns:.0f} ns ({metrics.total_length_ns / 1000:.1f} us)"
)

op_ir = run_pulse_to_operator(program, target=target)
print(f"\nOperator program: {len(op_ir.hamiltonian_terms)} Hamiltonian terms, "
      f"{len(op_ir.dissipator_terms)} dissipator terms")
