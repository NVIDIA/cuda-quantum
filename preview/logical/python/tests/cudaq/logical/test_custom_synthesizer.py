# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# Licensed under the Apache License, Version 2.0.

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

import cudaq
import cudaq.logical as cql
from cudaq.logical.compiler import CliffordTWord
from cudaq.logical.compiler.synthesizers import _verify_word


@cql.program
def repeated_rz() -> bool:
    q = cql.prepare_zero()
    q = cql.rz(q, 0.1234)
    q = cql.rz(q, 0.1234)
    return cql.measure_z(q)


def test_rz_dispatch_cache_provenance_and_determinism(synthesizer_factory):
    synth = synthesizer_factory()
    first = cql.compiler.synthesize(repeated_rz,
                                    gate_set=cql.compiler.clifford_t,
                                    precision=1e-6,
                                    synthesizer=synth)
    assert synth.calls == [("rz", 0.1234, 1e-6)]
    assert first.synthesis.synthesizer == "test-synth/1"
    record = next(item for item in first.evidence
                  if item.kind == "logical_gate_set_legalization")
    assert record.producer == "test-synth/1"
    second = cql.compiler.synthesize(repeated_rz,
                                     gate_set=cql.compiler.clifford_t,
                                     precision=1e-6,
                                     synthesizer=synth)
    assert len(synth.calls) == 2  # Cache belongs to a pass run.
    assert first.to_mlir() == second.to_mlir()
    assert "pauli_rotation" not in first.to_mlir()


def test_pipeline_compile_propagates_synthesizer(synthesizer_factory):
    synth = synthesizer_factory()
    pipeline = cql.compiler.pipelines.clifford_t(precision=1e-6,
                                                 synthesizer=synth)
    build = cql.compile(repeated_rz, pipeline=pipeline)
    assert synth.calls == [("rz", 0.1234, 1e-6)]
    assert build.synthesis.synthesizer == "test-synth/1"
    assert pipeline == cql.compiler.clifford_t.pipeline(precision=1e-6,
                                                        synthesizer=synth)


def test_float_rz_lattice_shortcut_respects_precision(synthesizer_factory):
    synth = synthesizer_factory()
    angle = float(cql.pi / 4)

    @cql.program
    def circuit() -> bool:
        return cql.measure_z(cql.rz(cql.prepare_zero(), angle))

    cql.compiler.synthesize(circuit,
                            gate_set=cql.compiler.clifford_t,
                            precision=1e-20,
                            synthesizer=synth)
    assert synth.calls == [("rz", angle, 1e-20)]


def test_same_names_have_independent_registrations(synthesizer_factory):
    first, second = synthesizer_factory(), synthesizer_factory()
    for synth in (first, second, first):
        cql.compiler.synthesize(repeated_rz,
                                gate_set=cql.compiler.clifford_t,
                                precision=1e-6,
                                synthesizer=synth)
    assert len(first.calls) == 2
    assert len(second.calls) == 1


@pytest.mark.parametrize("via_target", [False, True])
@pytest.mark.parametrize("replay", [False, True])
def test_already_synthesized_build_preserves_provenance(synthesizer_factory,
                                                        via_target, replay):
    first, other = synthesizer_factory(), synthesizer_factory()
    build = cql.compiler.synthesize(repeated_rz,
                                    gate_set=cql.compiler.clifford_t,
                                    precision=1e-6,
                                    synthesizer=first)
    if replay:
        build = cql.compiler.Build.replay(build.serialize())

    def apply(synthesizer, precision):
        if via_target:
            backend = cql.targets.clifford_t_target(
                precision=precision,
                synthesizer=synthesizer).runtime_endpoint.next_backend
            return backend.compile(build)
        return cql.compiler.synthesize(build,
                                       gate_set=cql.compiler.clifford_t,
                                       precision=precision,
                                       synthesizer=synthesizer)

    for precision in (1e-6, 1e-4):
        assert apply(first, precision) is build
    for synthesizer, precision in ((other, 1e-6), (None, 1e-6), (first, 1e-8)):
        with pytest.raises(ValueError, match="requires the original program"):
            apply(synthesizer, precision)
    assert first.calls == [("rz", 0.1234, 1e-6)]
    assert other.calls == []


@pytest.mark.parametrize("via_target", [False, True])
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("transform", ["to_pbc", "absorb_clifford_frame"])
def test_transformed_build_cannot_relabel_synthesis(synthesizer_factory,
                                                    via_target, replay,
                                                    transform):

    @cql.program
    def circuit() -> None:
        q = cql.rz(cql.prepare_zero(), 0.1234)
        cql.discard(q)

    first = cql.compiler.synthesize(circuit,
                                    gate_set=cql.compiler.clifford_t,
                                    precision=1e-6)
    build = cql.compiler.to_pbc(first)
    if transform == "absorb_clifford_frame":
        build = cql.compiler.absorb_clifford_frame(build)
    if replay:
        build = cql.compiler.Build.replay(build.serialize())
    assert build.synthesis is None
    assert any(record.kind == "logical_gate_set_legalization"
               for record in build.evidence)
    before = build.serialize()
    other = synthesizer_factory()
    for precision, synthesizer in ((1e-20, other), (1e-20, None), (1e-4, None)):
        with pytest.raises(ValueError, match="requires the original program"):
            if via_target:
                backend = cql.targets.clifford_t_target(
                    precision=precision,
                    synthesizer=synthesizer).runtime_endpoint.next_backend
                backend.compile(build)
            else:
                cql.compiler.synthesize(build,
                                        gate_set=cql.compiler.clifford_t,
                                        precision=precision,
                                        synthesizer=synthesizer)
    assert other.calls == []
    assert build.serialize() == before


def test_absorbed_unsynthesized_build_can_be_synthesized(synthesizer_factory):

    @cql.program
    def circuit() -> None:
        q = cql.rz(cql.prepare_zero(), 0.1234)
        cql.discard(q)

    build = cql.compiler.absorb_clifford_frame(cql.compile(circuit))
    synth = synthesizer_factory()
    result = cql.compiler.synthesize(build,
                                     gate_set=cql.compiler.clifford_t,
                                     precision=1e-6,
                                     synthesizer=synth)
    assert synth.calls == [("rz", 0.1234, 1e-6)]
    assert result.synthesis.synthesizer == "test-synth/1"


@pytest.mark.parametrize("gates,angle,expected", [
    (("sdg",), -float(cql.pi / 2), ("s",) * 3),
    (("tdg",), -float(cql.pi / 4), ("s", "s", "s", "t")),
    (("x",), float(cql.pi), ("h", "s", "s", "h")),
])
def test_alphabet_normalization(gates, angle, expected):
    kind = "u3" if gates == ("x",) else "rz"
    angles = (angle, -float(cql.pi / 2),
              float(cql.pi / 2)) if kind == "u3" else (angle,)
    word = CliffordTWord(gates, 1e-6, 0, "test", "test", "1")
    assert _verify_word(kind, angles, 1e-6, word).gates == expected


@pytest.mark.parametrize("gates,message", [
    (("cx",), "unsupported Clifford\\+T gate"),
    ((), "exceeds requested precision"),
])
def test_invalid_plugin_word_fails_compilation(synthesizer_factory, gates,
                                               message):
    synth = synthesizer_factory()
    synth.rz = lambda angle, eps: CliffordTWord(gates, eps, 0, "test", "test",
                                                "1")
    with pytest.raises(RuntimeError, match=message):
        cql.compiler.synthesize(repeated_rz,
                                gate_set=cql.compiler.clifford_t,
                                precision=1e-6,
                                synthesizer=synth)


def test_external_exception_has_original_message(synthesizer_factory):
    synth = synthesizer_factory()

    def fail(angle, precision):
        raise ValueError("external factoring budget exhausted")

    synth.rz = fail
    with pytest.raises(RuntimeError,
                       match="external factoring budget exhausted"):
        cql.compiler.synthesize(repeated_rz,
                                gate_set=cql.compiler.clifford_t,
                                precision=1e-6,
                                synthesizer=synth)


def test_double_precision_floor_is_not_silently_relaxed(monkeypatch):
    monkeypatch.setitem(sys.modules, "mpmath", None)
    identity = CliffordTWord((), 1e-6, 0, "test", "test", "1")
    assert _verify_word("rz", (0.0,), 1e-6, identity).gates == ()
    with pytest.raises(ValueError, match="install mpmath"):
        _verify_word("rz", (0.0,), 1e-16, identity)
    with pytest.raises(ValueError, match="exceeds requested precision"):
        _verify_word("rz", (0.1,), 1e-6, identity)


def test_high_precision_word_check():
    pytest.importorskip("mpmath")
    word = CliffordTWord(("h", "h"), 1e-25, 0, "test", "test", "1")
    assert _verify_word("rz", (0.0,), 1e-25, word).achieved_error < 1e-25
    with pytest.raises(ValueError, match="achieved_error"):
        _verify_word("rz", (0.0,), 1e-6,
                     replace(word, achieved_error=float("nan")))


def test_unknown_or_unsafe_registry_name():
    with pytest.raises(RuntimeError,
                       match="unknown rotation synthesizer 'missing'"):
        cql.compiler.synthesize(repeated_rz,
                                gate_set=cql.compiler.clifford_t,
                                synthesizer="missing")
    with pytest.raises(ValueError, match="invalid characters"):
        cql.compiler.clifford_t.pipeline(synthesizer="evil} , canonicalize")


def test_old_word_import_is_same_class():
    from cudaq.logical.qec.synthesis import CliffordTWord as OldWord
    assert OldWord is CliffordTWord


def test_native_registration_cannot_replace_builtin():
    from cudaq.mlir._mlir_libs import _qlxRuntime as runtime
    with pytest.raises(ValueError, match="already registered"):
        runtime.register_rotation_synthesizer("gridsynth", "test",
                                              lambda *args: None)


def test_pygridsynth_adapter():
    pytest.importorskip("pygridsynth")
    from cudaq.logical.compiler import PyGridSynth
    synth = PyGridSynth()
    build = cql.compiler.synthesize(repeated_rz,
                                    gate_set=cql.compiler.clifford_t,
                                    precision=1e-6,
                                    synthesizer=synth)
    assert build.synthesis.synthesizer.startswith("pygridsynth/")
    assert build.synthesis.t_count > 0


@pytest.mark.parametrize("angle", [0.1234, float(cql.pi / 4)])
def test_pygridsynth_uses_binary_float_at_tight_precision(angle):
    pytest.importorskip("pygridsynth")
    from cudaq.logical.qec.synthesis import pygridsynth_rz
    word = pygridsynth_rz(angle, 1e-20)
    checked = _verify_word("rz", (angle,), 1e-20, word)
    assert checked.achieved_error <= 1e-20
    assert word.algorithm == "ross-selinger-gridsynth"


def test_custom_synthesizer_generates_qec_protocol(synthesizer_factory):
    from types import SimpleNamespace
    from cudaq.logical.qec.lowering import ActionSiteHandle, QECCompilerContext
    from cudaq.logical.qec.synthesis import synthesis_strategy
    from cudaq.logical.qec import product_rotation

    @cql.code
    class Unencoded:
        block = cql.codes.CSSBlock(data=1)
        d = 1
        hx = hz = ()
        lx = lz = ((0,),)

    @cql.gadget(implements=cql.logical.h)
    def physical_h(block: cql.patch[Unencoded]) -> cql.patch[Unencoded]:
        return cql.h(block.data)

    @cql.gadget(implements=cql.logical.s)
    def physical_s(block: cql.patch[Unencoded]) -> cql.patch[Unencoded]:
        return cql.s(block.data)

    @cql.gadget(implements=cql.logical.t)
    def physical_t(block: cql.patch[Unencoded]) -> cql.patch[Unencoded]:
        return cql.t(block.data)

    @cql.gadget(implements=cql.logical.cx)
    def physical_cx(
        a: cql.patch[Unencoded], b: cql.patch[Unencoded]
    ) -> tuple[cql.patch[Unencoded], cql.patch[Unencoded]]:
        return cql.cx(a.data, b.data)

    synth = synthesizer_factory()
    strategy = synthesis_strategy(synth,
                                  h=physical_h,
                                  s=physical_s,
                                  cx=physical_cx,
                                  t=physical_t)
    lowering = product_rotation.compiler(code=Unencoded, synthesis=strategy)
    site = ActionSiteHandle("rotation", "action", "pauli_product_rotation",
                            None, ("q0",), {
                                "angle": 0.1234,
                                "precision": 1e-6,
                                "x_mask": 1,
                                "z_mask": 1,
                                "sign": -1
                            }, 1, 1)
    context = QECCompilerContext(None, lowering, Unencoded,
                                 Unencoded.default_encoding, {},
                                 strategy.dependencies,
                                 (SimpleNamespace(block="q0", placement="q0"),))
    artifact = strategy(site, context)
    assert synth.calls == [("rz", -0.1234, 1e-6)]
    assert artifact.specialization["synthesis_implementation"] == "test-synth"
    assert artifact.definition.metadata["compiler"] == "synthesis.rpp"
    assert artifact.specialization["synthesis_achieved_error"] <= 1e-6
    assert "fabric.call" in cql.compile(artifact.definition).to_mlir()


def test_callbacks_are_cleaned_up_at_interpreter_exit():
    script = """
import _cudaq_logical_devpath
from cudaq.logical.compiler.synthesizers import _ensure_registered
class Synth:
    name = 'shutdown'
    version = '1'
    def rz(self, angle, precision):
        raise AssertionError('not called')
_ensure_registered(Synth())
"""
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    assert "leaked" not in result.stderr


def test_example_runs_without_optional_mpmath():
    example = Path(__file__).resolve(
    ).parents[4] / "examples/standalone/06_custom_synthesizer.py"
    script = """
import sys
sys.modules['mpmath'] = None
import _cudaq_logical_devpath
import runpy
runpy.run_path(sys.argv[1], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", script,
                             str(example)],
                            capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    assert "toy-su2/0.1.0" in result.stdout


@cudaq.kernel
def kernel_rz_for_plugin():
    q = cudaq.qvector(1)
    rz(0.1234, q[0])
    mz(q[0])


def test_estimation_target_dispatch(synthesizer_factory):
    synth = synthesizer_factory()
    target = cql.targets.clifford_t_target(precision=1e-6, synthesizer=synth)
    try:
        cudaq.set_target(target)
        estimate = cudaq.estimate(kernel_rz_for_plugin)
    finally:
        cudaq.reset_target()
    assert len(synth.calls) == 1
    assert synth.calls[0][0] == "rz"
    assert estimate is not None
    assert ("synthesizer", "test-synth/1"
           ) in target.runtime_endpoint.next_backend.stack_metadata()
