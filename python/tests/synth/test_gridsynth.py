# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import math

import pytest

synth = pytest.importorskip(
    "cudaq.synth",
    reason="cudaq.synth not built (requires GMP/MPFR and "
    "CUDAQ_ENABLE_CLIFFORD_T_SYNTHESIS=ON)",
)

GATE_ALPHABET = set("HSTXWI")


def _is_valid_gate_string(s: str) -> bool:
    return len(s) > 0 and all(c in GATE_ALPHABET for c in s)


@pytest.mark.parametrize(
    "theta,epsilon",
    [
        (0.5, 1e-4),
        (1.0, 1e-4),
        (math.pi / 4, 1e-6),
        (math.pi / 8, 1e-6),
        (math.pi / 32, 1e-6),
    ],
)
def test_gridsynth_returns_valid_gate_string(theta, epsilon):
    gates = synth.gridsynth(theta, epsilon)
    assert _is_valid_gate_string(gates)
    assert gates.count("T") > 0


def test_gridsynth_t_count_within_theoretical_bound():
    epsilon = 1e-10
    gates = synth.gridsynth(math.pi / 4, epsilon)
    bound = math.ceil(4.0 * math.log2(1.0 / epsilon)) + 20
    assert gates.count("T") <= bound


def test_gridsynth_length_scales_with_precision():
    coarse = synth.gridsynth(math.pi / 4, 1e-4)
    fine = synth.gridsynth(math.pi / 4, 1e-10)
    assert fine.count("T") >= coarse.count("T")


def test_gridsynth_accepts_string_inputs():
    gates = synth.gridsynth("0.39269908169872414", "1e-15")
    assert _is_valid_gate_string(gates)


def test_gridsynth_rejects_nonpositive_epsilon():
    with pytest.raises(ValueError):
        synth.gridsynth(1.0, 0.0)
    with pytest.raises(ValueError):
        synth.gridsynth(1.0, -1e-3)


def test_gridsynth_keyword_timeouts():
    gates = synth.gridsynth(
        math.pi / 4,
        1e-6,
        diophantine_timeout_ms=500,
        factoring_timeout_ms=100,
    )
    assert _is_valid_gate_string(gates)
