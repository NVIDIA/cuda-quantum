# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Waveform gallery: every envelope type and waveform algebra.

Demonstrates all 8 built-in waveform constructors and the 5 algebraic
combinators (add, sub, mul, scale, neg) available inside a pulse kernel.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import cudaq_pulse as pulse

# ── 1. Every waveform constructor ────────────────────────────────────────────


@pulse.kernel
def waveform_showcase():
    """Drive a qubit with each built-in waveform type sequentially."""
    qubit = pulse.qudit_ref()
    drive_line, tone = get_drive_line(qubit)

    # Gaussian — the workhorse of single-qubit gates
    drive(drive_line, gaussian(40, 0.3, 10.0), tone)

    # Square — constant amplitude, simplest envelope
    drive(drive_line, square(100, 0.5), tone)

    # DRAG — derivative removal by adiabatic gate, reduces leakage
    drive(drive_line, drag(40, 0.25, 10.0, 0.5), tone)

    # Cosine — smooth rise/fall for flux pulses
    drive(drive_line, cosine(60, 0.4, 0.1), tone)

    # Tanh ramp — sigmoidal edges for adiabatic state transfer
    drive(drive_line, tanh_ramp(80, 0.35, 5.0), tone)

    # Gaussian square — flat top with Gaussian rise/fall, used in echoed-CR
    drive(drive_line, gaussian_square(200, 0.1, 10.0, 160), tone)

    # Custom — named envelope resolved at runtime from calibration DB
    drive(drive_line, custom(40, "my_optimal_pulse"), tone)

    # Custom samples — arbitrary IQ envelope from pre-computed data
    drive(drive_line, custom_samples([0.1, 0.3, 0.5, 0.3, 0.1]), tone)


# ── 2. Waveform algebra ──────────────────────────────────────────────────────


@pulse.kernel
def waveform_algebra():
    """Combine waveforms using the built-in algebraic operations."""
    qubit = pulse.qudit_ref()
    drive_line, tone = get_drive_line(qubit)

    gaussian_envelope = gaussian(40, 0.3, 10.0)
    square_envelope = square(40, 0.1)

    # Addition — superpose two envelopes (e.g. DRAG = Gaussian + derivative)
    combined = wf_add(gaussian_envelope, square_envelope)
    drive(drive_line, combined, tone)

    # Subtraction — difference of envelopes
    diff = wf_sub(gaussian_envelope, square_envelope)
    drive(drive_line, diff, tone)

    # Multiplication — element-wise product (amplitude modulation)
    modulated = wf_mul(gaussian_envelope, square_envelope)
    drive(drive_line, modulated, tone)

    # Scale — multiply envelope by a scalar
    boosted = wf_scale(gaussian_envelope, 2.0)
    drive(drive_line, boosted, tone)

    # Negation — flip sign (180° phase flip in IQ)
    flipped = wf_neg(gaussian_envelope)
    drive(drive_line, flipped, tone)


# ── 3. Composing complex envelopes ───────────────────────────────────────────


@pulse.kernel
def derivative_pulse(sigma, amplitude, beta):
    """DRAG-like pulse built manually: Gaussian + beta * d/dt(Gaussian).

    Shows how waveform algebra composes arbitrary envelopes from primitives.
    """
    qubit = pulse.qudit_ref()
    drive_line, tone = get_drive_line(qubit)
    dur = 40

    base = gaussian(dur, amplitude, sigma)
    derivative_approx = wf_sub(
        gaussian(dur, amplitude, sigma),
        gaussian(dur, amplitude, sigma),
    )
    correction = wf_scale(derivative_approx, beta)
    final_envelope = wf_add(base, correction)

    drive(drive_line, final_envelope, tone)


if __name__ == "__main__":
    print("=== Waveform Showcase ===")
    compiled_kernel = pulse.compile(waveform_showcase, [],
                                    qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== Waveform Algebra ===")
    compiled_kernel = pulse.compile(waveform_algebra, [],
                                    qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)

    print("\n=== Derivative Pulse ===")
    compiled_kernel = pulse.compile(derivative_pulse, [10.0, 0.3, 0.5],
                                    qubit_freq_hz={0: 5e9})
    print(compiled_kernel.mlir)
    print(f"  Compile metrics: {compiled_kernel.metrics}")
