# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full compilation pipeline walkthrough.

Demonstrates the ``pulse.compile()`` API — the single entry point that
captures a kernel, runs all optimisation passes, and produces a
``pulse.CompiledKernel`` containing lowered MLIR.
"""

import math

import cudaq_pulse as pulse


@pulse.kernel
def pipeline_demo(qubit_0, qubit_1):
    """A 2-qubit program with redundancies for the pipeline to clean up.

    Includes:
      - A shift_phase before a drive (virtual-Z folds it)
      - Adjacent waits (canonicalize merges them)
      - Adjacent same-amplitude square pulses (fusion merges them)
    """
    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    shift_phase(tone_0, math.pi / 4)
    gaussian_envelope = gaussian(40, 0.3, 10.0)
    drive(drive_line_0, gaussian_envelope, tone_0)

    wait(drive_line_0, 100)
    wait(drive_line_0, 100)

    square_pulse_1 = square(50, 0.2)
    drive(drive_line_0, square_pulse_1, tone_0)
    square_pulse_2 = square(50, 0.2)
    drive(drive_line_0, square_pulse_2, tone_0)

    gaussian_envelope_2 = gaussian(40, 0.3, 10.0)
    drive(drive_line_1, gaussian_envelope_2, tone_1)


def main():
    compiled_kernel = pulse.compile(
        pipeline_demo,
        [pulse.qudit_ref(), pulse.qudit_ref()],
        clock_ghz=1.0,
        qubit_freq_hz={
            0: 5.0e9,
            1: 5.1e9
        },
        schedule="alap",
    )

    print("=== Compiled MLIR (first 30 lines) ===")
    for line in compiled_kernel.mlir.splitlines()[:30]:
        print(f"  {line}")

    metrics = compiled_kernel.metrics
    print(f"\n=== Compile metrics ===")
    print(f"  Capture   : {metrics.capture_ms:.3f} ms")
    print(f"  Lower     : {metrics.lower_ms:.3f} ms")
    print(f"  Passes    : {metrics.passes_ms:.3f} ms")
    print(f"  Schedule  : {metrics.schedule_ms:.3f} ms")
    print(f"  MLIR emit : {metrics.mlir_emit_ms:.3f} ms")
    print(f"  Total     : {metrics.total_ms:.3f} ms")

    # Lower to LLVM IR (requires mlir-opt / cudaq-pulse-opt on PATH)
    try:
        llvm_ir = compiled_kernel.lower_to_llvm()
        print(f"\n=== LLVM IR ({len(llvm_ir)} chars, first 10 lines) ===")
        for line in llvm_ir.splitlines()[:10]:
            print(f"  {line}")
    except FileNotFoundError:
        print(
            "\n(Skipping LLVM lowering — mlir-opt / cudaq-pulse-opt not found)")


if __name__ == "__main__":
    main()
