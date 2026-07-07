# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import math

import pytest

from cudaq import synth

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
def test_gridsynth_returns_valid_sequence(theta, epsilon):
    seq = synth.gridsynth(theta, epsilon)
    assert isinstance(seq, synth.CliffordTSequence)
    assert _is_valid_gate_string(str(seq))
    assert seq.t_count > 0


def test_gridsynth_t_count_within_theoretical_bound():
    epsilon = 1e-10
    seq = synth.gridsynth(math.pi / 4, epsilon)
    bound = math.ceil(4.0 * math.log2(1.0 / epsilon)) + 20
    assert seq.t_count <= bound


def test_gridsynth_length_scales_with_precision():
    coarse = synth.gridsynth(math.pi / 4, 1e-4)
    fine = synth.gridsynth(math.pi / 4, 1e-10)
    assert fine.t_count >= coarse.t_count


def test_gridsynth_accepts_string_inputs():
    seq = synth.gridsynth("0.39269908169872414", "1e-15")
    assert _is_valid_gate_string(str(seq))


def test_gridsynth_rejects_nonpositive_epsilon():
    with pytest.raises(ValueError):
        synth.gridsynth(1.0, 0.0)
    with pytest.raises(ValueError):
        synth.gridsynth(1.0, -1e-3)


@pytest.mark.parametrize("bad", ["", "abc", "1.2.3", "1e", "0x10"])
def test_gridsynth_rejects_malformed_numeric_strings(bad):
    with pytest.raises(ValueError, match="invalid numeric string"):
        synth.gridsynth(bad, 1e-4)
    with pytest.raises(ValueError, match="invalid numeric string"):
        synth.gridsynth(1.0, bad)


def test_gridsynth_rejects_nonfinite_theta():
    for bad in (math.nan, math.inf, "nan", "inf"):
        with pytest.raises(ValueError, match="finite"):
            synth.gridsynth(bad, 1e-4)


def test_gridsynth_keyword_timeouts():
    seq = synth.gridsynth(
        math.pi / 4,
        1e-6,
        diophantine_timeout_ms=500,
        factoring_timeout_ms=100,
    )
    assert _is_valid_gate_string(str(seq))


def test_cliffordt_sequence_str_and_sequence_protocol():
    seq = synth.gridsynth(math.pi / 4, 1e-4)
    s = str(seq)
    assert repr(seq) == f"CliffordTSequence('{s}')"
    assert len(seq) == len(s)
    assert list(seq) == list(s)
    assert seq[0] == s[0]
    assert seq == s
    assert seq == synth.CliffordTSequence(s)


def test_cliffordt_sequence_identity():
    identity = synth.CliffordTSequence("I")
    assert str(identity) == "I"
    assert len(identity) == 0
    assert identity.t_count == 0
    assert list(identity) == []


def test_cliffordt_sequence_rejects_invalid_characters():
    with pytest.raises(ValueError):
        synth.CliffordTSequence("HTQ")


def test_cliffordt_sequence_t_count():
    seq = synth.CliffordTSequence("HTSHTW")
    assert seq.t_count == 2


def test_cliffordt_sequence_to_kernel_approximates_rz():
    cudaq = pytest.importorskip("cudaq")
    np = pytest.importorskip("numpy")

    theta, epsilon = math.pi / 8, 1e-6
    seq = synth.gridsynth(theta, epsilon)
    seq_kernel = seq.to_kernel()

    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.h(qubit)
    kernel.apply_call(seq_kernel, qubit)
    approx = np.array(cudaq.get_state(kernel))

    ref_kernel = cudaq.make_kernel()
    ref_qubit = ref_kernel.qalloc()
    ref_kernel.h(ref_qubit)
    ref_kernel.rz(theta, ref_qubit)
    reference = np.array(cudaq.get_state(ref_kernel))

    overlap = abs(np.vdot(reference, approx))
    assert overlap > 1.0 - 1e-4


def test_cliffordt_sequence_identity_kernel():
    cudaq = pytest.importorskip("cudaq")

    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.apply_call(synth.CliffordTSequence("I").to_kernel(), qubit)
    counts = cudaq.sample(kernel)
    assert counts["0"] == sum(counts.values())
