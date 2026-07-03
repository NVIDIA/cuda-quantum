#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rabi oscillation on qubit D5 using the Krinner 17-qubit target.

Demonstrates:
  - Loading the pre-defined transmon target
  - Using per-qubit drive parameters from the target
  - Running verify + schedule on a single-qubit pulse program
  - Lowering to an operator program with the target Hamiltonian

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
qubit_idx = 4  # D5 -- central data qubit

qubit_d5 = target.qubits[qubit_idx]
drive_params = target.get_drive_params(qubit_idx)

print(f"Target: {target.name} ({target.n_qubits} qubits)")
print(f"Qubit D5 (idx {qubit_idx}):")
print(f"  frequency = {qubit_d5.frequency_hz / 1e9:.3f} GHz")
print(f"  anharmonicity = {qubit_d5.anharmonicity_hz / 1e6:.1f} MHz")
print(f"  T1 = {qubit_d5.t1_us:.1f} us, T2* = {qubit_d5.t2_star_us:.1f} us")
print(f"  DRAG params: amp={drive_params['x_amp']:.3f}, "
      f"sigma={drive_params['x_sigma']:.1f}, "
      f"beta={drive_params['x_beta']:.2f}")

amplitude = drive_params["x_amp"]
sigma = drive_params["x_sigma"]
beta = drive_params["x_beta"]
duration = drive_params["x_dur"]
n_rabi_points = 21


@pulse.kernel
def rabi_d5(qubit):
    drive_line, tone = get_drive_line(qubit)
    for i in range(n_rabi_points):
        scale = i / (n_rabi_points - 1)
        rabi_pulse = drag(duration, amplitude * scale, sigma, beta)
        drive(drive_line, rabi_pulse, tone)
        wait(drive_line, 100)


ir = rabi_d5(pulse.qudit_ref())
program = _to_program(ir,
                      clock_ghz=2.0,
                      qubit_freq_hz={qubit_idx: qubit_d5.frequency_hz})

issues = verify(program)
print(f"\nVerification: {len(issues)} issue(s)")
for issue in issues:
    print(f"  {issue}")

events, metrics = schedule_alap(program)
print(
    f"\nScheduled {metrics.op_count} ops, total length = {metrics.total_length_ns:.1f} ns"
)

op_ir = run_pulse_to_operator(program, target=target)
print(
    f"\nOperator program: {len(op_ir.hamiltonian_terms)} Hamiltonian terms, "
    f"{len(op_ir.dissipator_terms)} dissipator terms, {op_ir.n_qubits} qubits")
print(f"Total simulated time: {op_ir.total_time_ns:.1f} ns")
