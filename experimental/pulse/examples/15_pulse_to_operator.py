# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pulse-to-operator lowering and the cuDensityMat simulation pipeline.

Demonstrates the full lowering path from pulse IR to the quantum operator
(qop) dialect, including:
  - Static Hamiltonian terms (qubit frequencies -> sigma_z)
  - Time-dependent drive controls (drive ops -> callbacks)
  - Lindblad dissipators from T1/T2 calibration data

NOTE: This example uses the advanced internal API (``_to_program``) because
``run_pulse_to_operator`` operates on ``Program`` objects — a specialized
pass not available in the standard C++ compilation pipeline.
"""

import math

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program
from cudaq_pulse.passes.pulse_to_operator import (
    run_pulse_to_operator,
    OperatorProgram,
    OperatorTerm,
)


@pulse.kernel
def single_qubit_drive(qubit_0):
    """Simple single-qubit X drive for lowering."""
    drive_line_0, tone_0 = get_drive_line(qubit_0)
    waveform = gaussian(40, 0.3, 10.0)
    drive(drive_line_0, waveform, tone_0)


@pulse.kernel
def two_qubit_cr(qubit_0, qubit_1):
    """Two-qubit program with cross-resonance for lowering."""
    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    waveform_x = drag(40, 0.25, 10.0, 0.5)
    waveform_cr = gaussian_square(300, 0.05, 10.0, 200.0)

    drive(drive_line_0, waveform_x, tone_0)
    drive(drive_line_0, waveform_cr, tone_1)
    drive(drive_line_1, waveform_x, tone_1)


def print_operator_program(op_prog: OperatorProgram) -> None:
    """Pretty-print an operator program."""
    print(f"  Name: {op_prog.name}")
    print(f"  Qubits: {op_prog.n_qubits}")
    print(f"  Total time: {op_prog.total_time_ns:.1f} ns")
    print(f"  Ops emitted: {len(op_prog.ops)}")

    print(f"\n  Hamiltonian terms ({len(op_prog.hamiltonian_terms)}):")
    for term in op_prog.hamiltonian_terms:
        td = "time-dep" if term.time_dependent else "static"
        print(f"    {term.kind:20s}  qubits={term.qubit_indices}  "
              f"coeff={term.coefficient:.6g}  [{td}]")

    if op_prog.dissipator_terms:
        print(f"\n  Dissipator terms ({len(op_prog.dissipator_terms)}):")
        for term in op_prog.dissipator_terms:
            print(f"    {term.kind:20s}  qubits={term.qubit_indices}  "
                  f"gamma={term.coefficient:.6g}")


def main():
    # 1. Single-qubit drive -> operator (no dissipators)
    print("=" * 60)
    print("  Single-qubit drive -> Hamiltonian")
    print("=" * 60)
    ir = single_qubit_drive(pulse.qudit_ref())
    prog = _to_program(ir, clock_ghz=1.0, qubit_freq_hz={0: 5.0e9})
    op_prog = run_pulse_to_operator(prog)
    print_operator_program(op_prog)

    # 2. Same program with T1/T2 dissipators
    print(f"\n{'=' * 60}")
    print("  Single-qubit with T1=50us, T2=30us dissipators")
    print("=" * 60)
    op_prog_diss = run_pulse_to_operator(
        prog,
        t1_times={0: 50e3},
        t2_times={0: 30e3},
    )
    print_operator_program(op_prog_diss)

    # 3. Two-qubit cross-resonance -> operator
    print(f"\n{'=' * 60}")
    print("  Two-qubit CR -> Hamiltonian + dissipators")
    print("=" * 60)
    ir_cr = two_qubit_cr(pulse.qudit_ref(), pulse.qudit_ref())
    prog_cr = _to_program(ir_cr,
                          clock_ghz=1.0,
                          qubit_freq_hz={
                              0: 5.0e9,
                              1: 5.15e9
                          })
    op_prog_cr = run_pulse_to_operator(
        prog_cr,
        t1_times={
            0: 80e3,
            1: 60e3
        },
        t2_times={
            0: 50e3,
            1: 40e3
        },
    )
    print_operator_program(op_prog_cr)

    # 4. Show the op-level IR for inspection
    print(f"\n{'=' * 60}")
    print("  Generated qop IR (two-qubit)")
    print("=" * 60)
    for i, op in enumerate(op_prog_cr.ops):
        res_names = [v.name for v in op.results if v.name]
        print(f"  [{i:2d}] {op.kind:30s} -> {res_names}")


if __name__ == "__main__":
    main()
