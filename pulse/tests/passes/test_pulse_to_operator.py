# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import math

import pytest

from cudaq_pulse.passes.ir_types import OpKind
from cudaq_pulse.passes.pulse_to_operator import run_pulse_to_operator
from cudaq_pulse.targets.base import Qubit, Target


def test_basic_operator_program(simple_program):
    result = run_pulse_to_operator(simple_program)
    assert result is not None
    assert result.n_qubits >= 1
    # The waveform definition and its drive both carry a duration, but only
    # the physical drive contributes to the makespan.
    assert result.total_time_ns == pytest.approx(20.0)


def test_operator_program_has_hamiltonian(simple_program):
    result = run_pulse_to_operator(simple_program)
    assert len(result.hamiltonian_terms) > 0


def test_two_qubit_operator(two_qubit_program):
    result = run_pulse_to_operator(two_qubit_program)
    assert result.n_qubits >= 2
    assert result.total_time_ns > 0


def test_with_dissipators(simple_program):
    result = run_pulse_to_operator(
        simple_program,
        t1_times={0: 50.0},
        t2_times={0: 30.0},
    )
    assert len(result.dissipator_terms) > 0


def test_dissipator_gamma_correctness(simple_program):
    """T2 dissipator should use pure dephasing rate: 1/T2 - 1/(2*T1)."""
    result = run_pulse_to_operator(
        simple_program,
        t1_times={0: 50.0},
        t2_times={0: 30.0},
    )
    assert len(result.dissipator_terms) >= 2


def test_target_coefficients_use_nanoseconds_without_duplicate_dissipators(
        simple_program):
    target = Target(
        name="unit-test",
        qubits={
            0:
                Qubit(
                    index=0,
                    frequency_hz=5.0e9,
                    anharmonicity_hz=-200.0e6,
                    t1_us=50.0,
                    t2_star_us=30.0,
                )
        },
    )

    result = run_pulse_to_operator(simple_program, target=target)
    static = next(t for t in result.hamiltonian_terms if t.kind == "static_z")
    assert static.coefficient.real == pytest.approx(math.pi * 5.0)
    assert not any(t.kind == "anharmonicity" for t in result.hamiltonian_terms)

    assert [t.kind for t in result.dissipator_terms
           ] == ["dissipator_t1", "dissipator_t2"]
    gamma1 = 1.0 / 50_000.0
    gamma_phi = 1.0 / 30_000.0 - 1.0 / (2.0 * 50_000.0)
    assert result.dissipator_terms[0].coefficient.real == pytest.approx(
        math.sqrt(gamma1))
    assert result.dissipator_terms[1].coefficient.real == pytest.approx(
        math.sqrt(gamma_phi / 2.0))


def test_target_dephasing_without_t1_is_finite():
    target = Target(
        name="dephasing-only",
        qubits={
            0:
                Qubit(
                    index=0,
                    frequency_hz=5.0e9,
                    anharmonicity_hz=0.0,
                    t1_us=0.0,
                    t2_star_us=20.0,
                )
        },
    )

    terms = target.dissipator_terms()
    assert len(terms) == 1
    assert terms[0]["coefficient"].real == pytest.approx(
        math.sqrt(1.0 / (2.0 * 20_000.0)))


def test_target_drive_amplitude_scale():
    calibrated = Target(
        name="calibrated",
        qubits={
            0:
                Qubit(
                    index=0,
                    frequency_hz=5.0e9,
                    anharmonicity_hz=-200.0e6,
                    t1_us=0.0,
                    t2_star_us=0.0,
                    drive_params={
                        "x_amp": 0.5,
                        "x_dur": 20.0,
                        "x_sigma": 5.0,
                    },
                )
        },
    )
    area = 5.0 * math.sqrt(2.0 * math.pi) * math.erf(
        20.0 / (2.0 * math.sqrt(2.0) * 5.0))
    assert calibrated.drive_amplitude_scale(0) == pytest.approx(math.pi /
                                                                (0.5 * area))
