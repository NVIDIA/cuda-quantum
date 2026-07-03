# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T1/T2 measurement with Lindblad dissipator.

Prepares a qubit, applies a wait period, and measures. Demonstrates
both the kernel frontend (via ``pulse.compile()``) and the direct IR builder
for pulse-to-operator lowering that adds Lindblad dissipators from
T1/T2 calibration data.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import cudaq_pulse as pulse


@pulse.kernel
def t1_experiment(wait_time):
    """T1 measurement: X pulse -> wait -> readout.

    The wait duration is parameterized to sweep T1 decay.
    """
    qubit = pulse.qudit_ref()
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    x_pulse = drag(40, 0.5, 10.0, 0.5)
    drive(drive_line, x_pulse, drive_tone)

    wait(drive_line, wait_time)

    sync(drive_line, readout_line)
    readout(readout_line, square(1000, 0.05), readout_tone)


@pulse.kernel
def t2_ramsey(tau):
    """T2* (Ramsey) measurement: pi/2 -> tau -> pi/2 -> readout."""
    qubit = pulse.qudit_ref()
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    half_pi = drag(40, 0.25, 10.0, 0.5)

    drive(drive_line, half_pi, drive_tone)
    wait(drive_line, tau)
    drive(drive_line, half_pi, drive_tone)

    sync(drive_line, readout_line)
    readout(readout_line, square(1000, 0.05), readout_tone)


@pulse.kernel
def t2_echo(tau):
    """T2 (Hahn echo) measurement: pi/2 -> tau/2 -> pi -> tau/2 -> pi/2 -> readout."""
    qubit = pulse.qudit_ref()
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    half_pi = drag(40, 0.25, 10.0, 0.5)
    pi_pulse = drag(40, 0.50, 10.0, 0.5)
    half_tau = tau / 2

    drive(drive_line, half_pi, drive_tone)
    wait(drive_line, half_tau)
    drive(drive_line, pi_pulse, drive_tone)
    wait(drive_line, half_tau)
    drive(drive_line, half_pi, drive_tone)

    sync(drive_line, readout_line)
    readout(readout_line, square(1000, 0.05), readout_tone)


if __name__ == "__main__":
    print("=== T1 experiment (sweep wait times) ===")
    for tau in [100, 500, 1000, 5000, 10000]:
        compiled_kernel = pulse.compile(t1_experiment, [tau],
                                        qubit_freq_hz={0: 5e9})
        print(f"  tau={tau:6d} VTU: compiled in "
              f"{compiled_kernel.metrics.total_ms:.3f} ms")

    print("\n=== T2* Ramsey (tau=500) ===")
    compiled_kernel = pulse.compile(t2_ramsey, [500], qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)

    print("\n=== T2 Echo (tau=1000) ===")
    compiled_kernel = pulse.compile(t2_echo, [1000], qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)

    print("\n=== Pulse-to-operator lowering with dissipators ===")
    from cudaq_pulse.passes.ir_types import (
        Program,
        Value,
        ValueType,
        Op,
        OpKind,
        _mk,
        _reset_vid_counter,
    )
    from cudaq_pulse.passes.pulse_to_operator import run_pulse_to_operator

    _reset_vid_counter()
    drive_line_0 = _mk(ValueType.DRIVE_LINE, "d0")
    tone_0 = _mk(ValueType.TONE, "t0")
    waveform = _mk(ValueType.WAVEFORM, "x_pulse")
    drive_line_1 = _mk(ValueType.DRIVE_LINE)
    tone_1 = _mk(ValueType.TONE)
    drive_line_2 = _mk(ValueType.DRIVE_LINE)

    program = Program(
        name="t1_lowering",
        clock_ghz=1.0,
        ops=[
            Op(OpKind.ALLOC_DRIVE, (), (drive_line_0, tone_0), {
                "qubit": 0,
                "freq_hz": 5.0e9
            }),
            Op(OpKind.MAKE_WAVEFORM, (), (waveform,), {
                "waveform_type": "drag",
                "duration_vtu": 40,
                "amplitude": 0.5
            }),
            Op(OpKind.DRIVE, (drive_line_0, waveform, tone_0),
               (drive_line_1, tone_1), {
                   "duration_vtu": 40,
                   "amplitude": 0.5,
                   "qubit_index": 0
               }),
            Op(OpKind.WAIT, (drive_line_1,), (drive_line_2,),
               {"duration_vtu": 500}),
        ],
        values=[drive_line_0, tone_0, waveform],
        qubit_freq_hz={0: 5.0e9},
    )

    op_prog = run_pulse_to_operator(program,
                                    t1_times={0: 50e3},
                                    t2_times={0: 30e3})
    print(f"  Hamiltonian terms: {len(op_prog.hamiltonian_terms)}")
    print(f"  Dissipator terms: {len(op_prog.dissipator_terms)}")
    for term in op_prog.dissipator_terms:
        print(
            f"    {term.kind}: qubit={term.qubit_indices}, gamma={term.coefficient:.4g}"
        )
