#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""End-to-end GPU time evolution on the Krinner 17-qubit target.

HARDWARE REQUIREMENT: This example requires an NVIDIA GPU with CUDA
runtime and cuDensityMat libraries installed. If unavailable, it will
raise an error with an actionable message -- no silent fallback.

Pipeline:
  @pulse.kernel -> pulse.evolve(target=...) -> native MLIR lowering
                -> JIT compile -> cuDensityMat GPU execution
"""

import math
import sys

import numpy as np

import cudaq_pulse as pulse
from cudaq_pulse.runtime.jit import _check_gpu_available
from cudaq_pulse.targets import transmon_krinner_17q

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

# This sequence is shorter than 128 ns. The remaining interval evolves under
# the target's always-on coupling and T1/T2 model.
result = pulse.evolve(
    ir,
    target=target,
    t_start=0.0,
    t_end=128.0,
    num_steps=512,
    integrator="rk4",
)

state = result.final_state
print(f"=== Final state shape: {state.shape} ===")
if state.ndim == 2:
    populations = np.real(np.diag(state))
    print(f"Density-matrix trace: {np.trace(state):.8f}")
else:
    populations = np.abs(state)**2

for basis_index, probability in enumerate(populations):
    if probability > 1.0e-6:
        print(f"  |{basis_index:02b}> : P = {probability:.6f}")
