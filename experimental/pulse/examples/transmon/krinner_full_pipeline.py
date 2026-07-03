#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full compilation pipeline demo on the Krinner target.

  kernel -> verify -> schedule -> pulse_to_operator(target) -> to_cudm_mlir

Prints the generated cuDensityMat MLIR text at the end.

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
from cudaq_pulse.passes.to_cudm_mlir import emit_cudm_mlir
from cudaq_pulse.targets import transmon_krinner_17q

target = transmon_krinner_17q()

qubit_0_info = target.qubits[0]
qubit_1_info = target.qubits[1]
drive_params_0 = target.get_drive_params(0)
drive_params_1 = target.get_drive_params(1)


@pulse.kernel
def krinner_bell(qubit_0, qubit_1):
    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    # Hadamard on Q0
    shift_phase(tone_0, math.pi / 2)
    sx_pulse = drag(20, drive_params_0["x_amp"], drive_params_0["x_sigma"],
                    drive_params_0["x_beta"])
    drive(drive_line_0, sx_pulse, tone_0)
    shift_phase(tone_0, math.pi / 2)

    # Sync
    sync(drive_line_0, drive_line_1)

    # Echoed CR (CZ-like)
    cr = gaussian(98, 0.32, 24.0)
    drive(drive_line_0, cr, tone_1)
    x_echo = drag(20, drive_params_1["x_amp"], drive_params_1["x_sigma"],
                  drive_params_1["x_beta"])
    drive(drive_line_1, x_echo, tone_1)
    cr_neg = gaussian(98, -0.32, 24.0)
    drive(drive_line_0, cr_neg, tone_1)
    drive(drive_line_1, x_echo, tone_1)


ir = krinner_bell(pulse.qudit_ref(), pulse.qudit_ref())
program = _to_program(
    ir,
    clock_ghz=2.0,
    qubit_freq_hz={
        0: qubit_0_info.frequency_hz,
        1: qubit_1_info.frequency_hz
    },
)

# Stage 1: Verify
print("=== Stage 1: Verify ===")
issues = verify(program)
print(f"  {len(issues)} issue(s)")

# Stage 2: Schedule
print("\n=== Stage 2: Schedule (ALAP) ===")
events, metrics = schedule_alap(program)
print(f"  {metrics.op_count} ops, {metrics.total_length_ns:.0f} ns total")

# Stage 3: Pulse-to-operator
print("\n=== Stage 3: Pulse-to-Operator ===")
op_ir = run_pulse_to_operator(program, target=target)
print(f"  {len(op_ir.hamiltonian_terms)} Hamiltonian terms")
print(f"  {len(op_ir.dissipator_terms)} dissipator terms")
print(f"  {op_ir.n_qubits} qubits, {op_ir.total_time_ns:.1f} ns total time")

# Stage 4: cuDensityMat MLIR emission
print("\n=== Stage 4: cuDensityMat MLIR ===")
cudm_mlir = emit_cudm_mlir(op_ir,
                           t_start=0.0,
                           t_end=op_ir.total_time_ns,
                           num_steps=100,
                           integrator="magnus_cf4")
print(cudm_mlir)
