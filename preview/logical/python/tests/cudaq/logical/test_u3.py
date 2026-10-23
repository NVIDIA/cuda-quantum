# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# Licensed under the Apache License, Version 2.0.

import math
from itertools import product

import numpy as np
import pytest

import cudaq.logical as cql


def _unitary(build):
    matrices = {
        "h": np.array([[1, 1], [1, -1]]) / math.sqrt(2),
        "s": np.diag([1, 1j]),
        "t": np.diag([1, np.exp(1j * math.pi / 4)]),
    }
    result = np.eye(2, dtype=complex)
    for line in build.to_mlir().splitlines():
        for gate, matrix in matrices.items():
            if f"#qlx.action<{gate}>" in line:
                result = matrix @ result
    return result


def _target(theta, phi, lam):

    def rz(angle):
        return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])

    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return rz(phi) @ np.array([[c, -s], [s, c]]) @ rz(lam)


def _assert_equivalent(build, theta, phi, lam, precision):
    actual, target = _unitary(build), _target(theta, phi, lam)
    phase = np.angle(np.trace(target.conj().T @ actual))
    assert np.linalg.norm(actual - np.exp(1j * phase) * target,
                          ord=2) <= precision


@pytest.mark.parametrize("direct", [False, True])
def test_u3_dispatch_precision_and_random_unitaries(synthesizer_factory,
                                                    direct):
    synth = synthesizer_factory(direct=direct)
    for theta, phi, lam in np.random.default_rng(123).uniform(-3, 3, (3, 3)):

        @cql.program
        def circuit() -> bool:
            q = cql.u3(cql.prepare_zero(),
                       float(theta),
                       float(phi),
                       float(lam),
                       precision=1e-6)
            return cql.measure_z(q)

        synth.calls.clear()
        build = cql.compiler.synthesize(circuit,
                                        gate_set=cql.compiler.clifford_t,
                                        precision=1e-4,
                                        synthesizer=synth)
        assert [call[0] for call in synth.calls
               ] == (["u3"] if direct else ["rz"] * 3)
        assert all(
            call[-1] == (1e-6 if direct else 1e-6 / 3) for call in synth.calls)
        _assert_equivalent(build, theta, phi, lam, 1e-6)


def test_default_u3_is_equivalent():

    @cql.program
    def circuit() -> bool:
        return cql.measure_z(cql.u3(cql.prepare_zero(), 0.3, 1.1, -0.4))

    build = cql.compiler.synthesize(circuit,
                                    gate_set=cql.compiler.clifford_t,
                                    precision=1e-6)
    _assert_equivalent(build, 0.3, 1.1, -0.4, 1e-6)


@pytest.mark.parametrize("mode", ["native", "split", "direct"])
@pytest.mark.parametrize("angle,precision", [(1e3, 1e-10), (1e6, 1e-10),
                                             (1e9, 1e-10), (1e16, 1e-6),
                                             (-1e-16, 1e-18)])
def test_numeric_angles_are_checked_against_original_f64(
        synthesizer_factory, mode, angle, precision):
    from cudaq.logical.compiler import CliffordTWord
    from cudaq.logical.compiler.synthesizers import _verify_word
    pytest.importorskip("mpmath")
    synth = (None if mode == "native" else synthesizer_factory(
        direct=mode == "direct"))

    @cql.program
    def circuit() -> bool:
        return cql.measure_z(cql.u3(cql.prepare_zero(), angle, 0.0, 0.0))

    build = cql.compiler.synthesize(circuit,
                                    gate_set=cql.compiler.clifford_t,
                                    precision=precision,
                                    synthesizer=synth)
    if synth is not None:
        # Both the whole-U3 callback and the split Rz callback see the raw
        # signed float, including small negatives that modulo would erase.
        assert synth.calls[0][1] == angle
    gates = tuple(
        line.split("#qlx.action<")[1].split(">")[0]
        for line in build.to_mlir().splitlines()
        if "#qlx.action<" in line)
    word = CliffordTWord(gates, precision, 0.0, "test", "test", "1")
    assert _verify_word("u3", (angle, 0.0, 0.0), precision,
                        word).achieved_error <= precision


def test_float_pi_is_not_exact_at_arbitrary_precision(synthesizer_factory):
    synth = synthesizer_factory(direct=True)

    @cql.program
    def circuit() -> bool:
        return cql.measure_z(cql.u3(cql.prepare_zero(), math.pi / 4, 0.0, 0.0))

    cql.compiler.synthesize(circuit,
                            gate_set=cql.compiler.clifford_t,
                            precision=1e-20,
                            synthesizer=synth)
    assert synth.calls == [("u3", math.pi / 4, 0.0, 0.0, 1e-20)]


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("quarter_turns", [1, 2, -2])
def test_mixed_exact_angles_keep_symbolic_target(synthesizer_factory, index,
                                                 quarter_turns):
    mp = pytest.importorskip("mpmath").mp.clone()
    mp.dps = 90
    angles = [0.1234, 0.2345, -0.3456]
    expected = [mp.mpf(angle) for angle in angles]
    angles[index] = quarter_turns * cql.pi / 4
    expected[index] = quarter_turns * mp.pi / 4
    synth = synthesizer_factory(direct=True)
    precision = 1e-20

    @cql.program
    def circuit() -> bool:
        return cql.measure_z(cql.u3(cql.prepare_zero(), *angles))

    build = cql.compiler.synthesize(circuit,
                                    gate_set=cql.compiler.clifford_t,
                                    precision=precision,
                                    synthesizer=synth)
    matrices = {
        "h": mp.matrix([[1, 1], [1, -1]]) / mp.sqrt(2),
        "s": mp.diag([1, mp.j]),
        "t": mp.diag([1, mp.exp(mp.j * mp.pi / 4)]),
    }
    actual = mp.eye(2)
    for line in build.to_mlir().splitlines():
        for gate, matrix in matrices.items():
            if f"#qlx.action<{gate}>" in line:
                actual = matrix * actual

    def rz(angle):
        return mp.diag([mp.exp(-mp.j * angle / 2), mp.exp(mp.j * angle / 2)])

    theta, phi, lam = expected
    c, s = mp.cos(theta / 2), mp.sin(theta / 2)
    target = rz(phi) * mp.matrix([[c, -s], [s, c]]) * rz(lam)
    overlap = sum(
        mp.conj(target[r, c]) * actual[r, c]
        for r in range(2)
        for c in range(2))
    delta = actual - mp.exp(mp.j * mp.arg(overlap)) * target
    # SVD checks the projective norm independently of the compiler's explicit
    # 2x2 norm formula, against symbolic pi rather than its rounded f64 proxy.
    assert max(mp.svd(delta, compute_uv=False)) <= mp.mpf(precision)
    assert len(synth.calls) == 2
    assert all(
        call[0] == "rz" and call[-1] == precision / 3 for call in synth.calls)


def test_native_lattice_check_handles_angles_beyond_int64():

    @cql.program
    def circuit() -> bool:
        return cql.measure_z(cql.u3(cql.prepare_zero(), 1e308, 0.0, 0.0))

    build = cql.compiler.synthesize(circuit,
                                    gate_set=cql.compiler.clifford_t,
                                    precision=1e-6)
    _assert_equivalent(build, 1e308, 0.0, 0.0, 1e-6)


@pytest.mark.parametrize("floating", [False, True])
def test_exact_h_is_one_h_without_plugin_calls(synthesizer_factory, floating):
    synth = synthesizer_factory(direct=True)
    theta = math.pi / 2 if floating else cql.pi / 2
    lam = math.pi if floating else cql.pi

    @cql.program
    def circuit() -> bool:
        return cql.measure_z(cql.u3(cql.prepare_zero(), theta, 0.0, lam))

    build = cql.compiler.synthesize(circuit,
                                    gate_set=cql.compiler.clifford_t,
                                    synthesizer=synth)
    assert build.synthesis.gate_count == build.synthesis.h_count == 1
    assert synth.calls == []


def test_all_exact_clifford_euler_combinations(synthesizer_factory):
    synth = synthesizer_factory(direct=True)
    for theta, phi, lam in product(range(4), repeat=3):

        @cql.program
        def circuit() -> bool:
            q = cql.u3(cql.prepare_zero(), theta * cql.pi / 2, phi * cql.pi / 2,
                       lam * cql.pi / 2)
            return cql.measure_z(q)

        build = cql.compiler.synthesize(circuit,
                                        gate_set=cql.compiler.clifford_t,
                                        synthesizer=synth)
        assert build.synthesis.t_count == 0
        _assert_equivalent(build, theta * math.pi / 2, phi * math.pi / 2,
                           lam * math.pi / 2, 1e-12)
    assert synth.calls == []


def test_exact_t_euler_angles_bypass_plugin(synthesizer_factory):
    synth = synthesizer_factory(direct=True)

    @cql.program
    def circuit() -> bool:
        q = cql.u3(cql.prepare_zero(), cql.pi / 4, 0.0, cql.pi / 4)
        return cql.measure_z(q)

    build = cql.compiler.synthesize(circuit,
                                    gate_set=cql.compiler.clifford_t,
                                    synthesizer=synth)
    _assert_equivalent(build, math.pi / 4, 0.0, math.pi / 4, 1e-12)
    assert build.synthesis.t_count == 2
    assert synth.calls == []


def test_bad_whole_u3_word_is_rejected(synthesizer_factory):
    from cudaq.logical.compiler import CliffordTWord
    synth = synthesizer_factory(direct=True)
    synth.u3 = lambda theta, phi, lam, eps: CliffordTWord(
        (), eps, 0, "bad", "test", "1")

    @cql.program
    def circuit() -> bool:
        return cql.measure_z(cql.u3(cql.prepare_zero(), 0.3, 1.1, 0.4))

    with pytest.raises(RuntimeError,
                       match="u3 synthesis word exceeds requested precision"):
        cql.compiler.synthesize(circuit,
                                gate_set=cql.compiler.clifford_t,
                                synthesizer=synth)


def test_exact_angles_are_canonical_and_counted():

    @cql.program
    def circuit() -> bool:
        return cql.measure_z(
            cql.u3(cql.prepare_zero(), -cql.pi / 2, 0.0, cql.pi * 5))

    build = cql.compile(circuit)
    text = build.to_mlir()
    assert "theta_pi_numer = 3" in text
    assert "theta_pi_denom = 2" in text
    assert "lambda_pi_numer = 1" in text
    assert cql.estimate.logical_counts(build).actions["qlx_standard_u3"] == 1
    assert cql.logical.u3.operands.q0.index == 0


def test_runtime_angles_require_specialization(synthesizer_factory):

    @cql.program
    def circuit(theta: float) -> bool:
        return cql.measure_z(cql.u3(cql.prepare_zero(), theta, 1.1, 0.4))

    authored = cql.compile(circuit)
    assert "#qlx.action<u3>" in authored.to_mlir()
    with pytest.raises(
            RuntimeError,
            match="specialize the program parameter before synthesis"):
        cql.compiler.synthesize(authored, gate_set=cql.compiler.clifford_t)
    specialized = cql.compiler.synthesize(circuit,
                                          gate_set=cql.compiler.clifford_t,
                                          parameters={"theta": 0.3},
                                          precision=1e-6,
                                          synthesizer=synthesizer_factory())
    _assert_equivalent(specialized, 0.3, 1.1, 0.4, 1e-6)


def test_u3_rejected_before_placement():

    @cql.program
    def circuit() -> bool:
        return cql.measure_z(cql.u3(cql.prepare_zero(), 0.3, 1.1, 0.4))

    with pytest.raises(ValueError,
                       match="u3 must be legalized before placement"):
        cql.compiler.place(circuit, device=None)


@pytest.mark.parametrize("angle", [float("inf"), float("nan"), True, "0.3"])
def test_invalid_angles_fail_authoring(angle):

    @cql.program
    def circuit() -> bool:
        return cql.measure_z(cql.u3(cql.prepare_zero(), angle, 0.0, 0.0))

    with pytest.raises(TypeError, match="u3 angles must be finite"):
        cql.compile(circuit)
