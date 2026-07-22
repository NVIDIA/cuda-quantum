#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end GPU time evolution on the Krinner 17-qubit target.

HARDWARE REQUIREMENT: This example requires an NVIDIA GPU with CUDA
runtime and cuDensityMat libraries installed. If unavailable, it will
raise an error with an actionable message -- no silent fallback.

Pipeline:
  @pulse.kernel -> verify -> schedule -> pulse_to_operator(target)
                -> cuDensityMat MLIR -> JIT compile -> GPU execute

NOTE: This example uses the advanced internal API (``_to_program``) because
``run_pulse_to_operator`` operates on ``Program`` objects — a specialized
pass not available in the standard C++ compilation pipeline.
"""

import math
import sys

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program
from cudaq_pulse.passes.verify import verify
from cudaq_pulse.passes.scheduling import schedule_alap
from cudaq_pulse.passes.pulse_to_operator import run_pulse_to_operator
from cudaq_pulse.passes.to_cudm_mlir import emit_cudm_mlir
from cudaq_pulse.targets import transmon_krinner_17q
from cudaq_pulse.runtime.jit import _check_gpu_available, JITCompiler

# -- Pre-flight: verify GPU is available --
if not _check_gpu_available():
    print("ERROR: No NVIDIA GPU detected.", file=sys.stderr)
    print("This example requires:", file=sys.stderr)
    print("  - NVIDIA GPU (compute capability >= 7.0)", file=sys.stderr)
    print("  - CUDA runtime (set CUDA_HOME if needed)", file=sys.stderr)
    print("  - cuDensityMat runtime (set CUDM_RUNTIME_LIB if needed)",
          file=sys.stderr)
    raise RuntimeError(
        "GPU required for evolve(). Install CUDA toolkit and cuDensityMat, "
        "then set CUDA_HOME and CUDM_RUNTIME_LIB environment variables.")

target = transmon_krinner_17q()

qubit_0_info = target.qubits[0]
qubit_1_info = target.qubits[1]
drive_params_0 = target.get_drive_params(0)
drive_params_1 = target.get_drive_params(1)


@pulse.kernel
def krinner_evolve(qubit_0, qubit_1):
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


ir = krinner_evolve(pulse.qudit_ref(), pulse.qudit_ref())
program = _to_program(
    ir,
    clock_ghz=2.0,
    qubit_freq_hz={
        0: qubit_0_info.frequency_hz,
        1: qubit_1_info.frequency_hz
    },
)

print("=== Stage 1: Verify ===")
issues = verify(program)
errors = [i for i in issues if i.severity == "error"]
if errors:
    for e in errors:
        print(f"  {e}")
print(f"  {len(issues)} issue(s), {len(errors)} error(s)")

print("\n=== Stage 2: Schedule (ALAP) ===")
events, metrics = schedule_alap(program)
print(f"  {metrics.op_count} ops, {metrics.total_length_ns:.0f} ns")

print("\n=== Stage 3: Pulse-to-Operator ===")
op_ir = run_pulse_to_operator(program, target=target)
print(f"  {len(op_ir.hamiltonian_terms)} Hamiltonian terms, "
      f"{len(op_ir.dissipator_terms)} dissipator terms")

print("\n=== Stage 4: cuDensityMat MLIR ===")
t_end = op_ir.total_time_ns
num_steps = 200
cudm_mlir = emit_cudm_mlir(op_ir,
                           t_start=0.0,
                           t_end=t_end,
                           num_steps=num_steps,
                           integrator="magnus_cf4")
print(f"  Generated {len(cudm_mlir.splitlines())} lines of MLIR")

print("\n=== Stage 5: JIT Compile & GPU Execute ===")
jit = JITCompiler()
so_path = jit.compile_mlir_to_so(cudm_mlir)
print(f"  Compiled to: {so_path}")

result = jit.load_and_run(so_path)
import numpy as np

state = np.array(result, dtype=np.complex128)
print(f"\n=== Final State Vector (dim={len(state)}) ===")
for i, amplitude in enumerate(state):
    prob = abs(amplitude)**2
    if prob > 1e-6:
        print(f"  |{i:0{op_ir.n_qubits}b}> : {amplitude:.6f}  (P = {prob:.4f})")

print(
    f"\nTotal probability: {sum(abs(amplitude)**2 for amplitude in state):.6f}")
