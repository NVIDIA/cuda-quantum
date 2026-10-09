# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Public immutable P0 Pauli-based-computation transform."""

from __future__ import annotations

import pytest

import cudaq.logical as qlx
import cudaq.mlir.ir as mlir_ir


@qlx.program
def exact_quarter_turns() -> tuple[bool, bool]:
    qubits = qlx.allocate(2)
    qubits[0] = qlx.h(qubits[0])
    qubits[0], qubits[1] = qlx.cx(qubits[0], qubits[1])
    (qubits[1],) = qlx.ops.rotate(qlx.types.Z(qubits[1]),
                                  angle=qlx.types.pi / 4)
    return qlx.measure_x(qubits[0]), qlx.measure_z(qubits[1])


@qlx.program
def unsynthesized_clifford_only() -> bool:
    qubit = qlx.allocate(1)[0]
    qubit = qlx.h(qubit)
    return qlx.measure_x(qubit)


@qlx.program
def folded_identity_residual_repeat() -> bool:
    qubit = qlx.allocate(1)[0]

    def body(_iteration, value):
        value = qlx.h(value)
        value = qlx.t(value)
        value = qlx.h(value)
        return (value,)

    qubit, = qlx.ops.repeat(1024, carries=(qubit,), body=body)
    return qlx.measure_z(qubit)


@qlx.program
def bounded_nonidentity_residual_repeat() -> bool:
    qubit = qlx.allocate(1)[0]

    def body(_iteration, value):
        value = qlx.h(value)
        value = qlx.t(value)
        return (value,)

    qubit, = qlx.ops.repeat(2, carries=(qubit,), body=body)
    return qlx.measure_z(qubit)


@qlx.program
def folded_zero_count_repeat() -> bool:
    qubit = qlx.allocate(1)[0]

    def body(_iteration, value):
        value = qlx.h(value)
        value = qlx.t(value)
        return (value,)

    qubit, = qlx.ops.repeat(0, carries=(qubit,), body=body)
    return qlx.measure_z(qubit)


@qlx.program
def authored_discard_reason() -> bool:
    qubits = qlx.allocate(2)
    qubits[0] = qlx.t(qubits[0])
    qlx.discard(qubits[0], reason="ancilla-cleanup")
    return qlx.measure_z(qubits[1])


def _synthesized():
    return qlx.compiler.synthesize(
        exact_quarter_turns,
        gate_set=qlx.compiler.gate_sets.clifford_t,
    )


def _walk(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


def test_public_pbc_transform_is_device_free_immutable_and_replayable():
    source = _synthesized()
    original = source.to_mlir()
    result = qlx.compiler.to_pbc(source)

    assert source.to_mlir() == original
    assert result.profile == "p0"
    assert result.stage == qlx.stages.P0
    text = result.to_mlir()
    assert text.count("#qlx.action<pauli_rotation>") == source.synthesis.t_count
    assert "#qlx.instrument<mpp>" in text
    for absorbed in ("#qlx.action<h>", "#qlx.action<s>", "#qlx.action<cx>"):
        assert absorbed not in text
    assert tuple(item.name for item in result.pipeline.passes) == (
        "qlx-to-pbc",
        "qlx-verify-pbc",
        "qlx-verify-p0",
    )
    assert any(item.kind == "pauli_based_computation_normalization"
               for item in result.evidence)
    normalization = tuple(
        item for item in result.evidence
        if item.kind == "pauli_based_computation_normalization")
    assert len(normalization) == 1
    assert (f"source-build-content-sha256={source.content_sha256}"
            in normalization[0].assumptions)
    assert "sha256:sha256:" not in repr(normalization[0].assumptions)

    replayed = qlx.compiler.Build.replay(result.serialize())
    assert replayed.to_mlir() == result.to_mlir()
    assert replayed.pipeline == result.pipeline
    assert replayed.evidence == result.evidence
    assert replayed.content_sha256 == result.content_sha256


def test_failed_pbc_transform_preserves_the_source_build(capfd):

    @qlx.program
    def unsupported_rotation_support() -> bool:
        q0, q1 = qlx.allocate(2)
        q0, q1 = qlx.cx(q0, q1)

        def body(_iteration, value):
            value = qlx.h(value)
            value = qlx.t(value)
            return (value,)

        (q0,) = qlx.ops.repeat(2, carries=(q0,), body=body)
        qlx.discard(q1)
        return qlx.measure_z(q0)

    source = qlx.compiler.synthesize(
        unsupported_rotation_support,
        gate_set=qlx.compiler.gate_sets.clifford_t,
    )
    original = source.to_mlir()
    original_module = str(source._fresh_module())

    # Normalization expands the repeat before support validation rejects it.
    with pytest.raises(RuntimeError, match="QLX PBC lowering failed"):
        qlx.compiler.to_pbc(source)

    assert "rotation support escapes the repeat carry set" in (
        capfd.readouterr().err)
    assert source.to_mlir() == original
    assert str(source._fresh_module()) == original_module


def test_pbc_transform_uses_the_authenticated_snapshot_not_cached_inspection():
    source = _synthesized()
    expected = qlx.compiler.to_pbc(source)
    pristine = source.to_mlir()
    cached = source.module
    t_apply = next(operation for operation in _walk(cached.operation)
                   if operation.name == "qlx.apply" and
                   str(operation.attributes["action"]) == "#qlx.action<t>")
    with cached.context:
        t_apply.attributes["action"] = mlir_ir.Attribute.parse(
            "#qlx.action<h>", context=cached.context)

    assert source.to_mlir() == pristine
    assert "#qlx.action<h>" in str(cached)
    actual = qlx.compiler.to_pbc(source)

    assert actual.to_mlir() == expected.to_mlir()
    evidence = next(item for item in actual.evidence
                    if item.kind == "pauli_based_computation_normalization")
    assert f"source-build-content-sha256={source.content_sha256}" in (
        evidence.assumptions)


def test_compile_pipeline_matches_pbc_convenience_transform():
    source = _synthesized()
    convenience = qlx.compiler.to_pbc(source)
    advanced = qlx.compile(source, pipeline=qlx.compiler.pipelines.pbc())
    assert advanced.to_mlir() == convenience.to_mlir()
    assert advanced.pipeline == convenience.pipeline


def test_pbc_dispatch_rejects_a_contradictory_pipeline_contract():
    source = _synthesized()
    contradictory = qlx.compiler.Pipeline(
        passes=qlx.compiler.pipelines.pbc().passes,
        output_profile="p4",
    )
    with pytest.raises(ValueError,
                       match="canonical cudaq.logical.compiler.pipelines.pbc"):
        qlx.compile(source, pipeline=contradictory)


def test_pbc_rejects_unsynthesized_or_downstream_input():
    raw = qlx.compile(exact_quarter_turns,
                      pipeline=qlx.compiler.pipelines.logical())
    with pytest.raises(ValueError, match="cudaq.logical.compiler.synthesize"):
        qlx.compiler.to_pbc(raw)
    raw_clifford = qlx.compile(
        unsynthesized_clifford_only,
        pipeline=qlx.compiler.pipelines.logical(),
    )
    with pytest.raises(ValueError, match="cudaq.logical.compiler.synthesize"):
        qlx.compiler.to_pbc(raw_clifford)
    with pytest.raises(TypeError, match="P0 Build"):
        qlx.compiler.to_pbc(exact_quarter_turns)


def test_pbc_preserves_an_exact_identity_residual_repeat():
    source = qlx.compiler.synthesize(
        folded_identity_residual_repeat,
        gate_set=qlx.compiler.gate_sets.clifford_t,
    )
    original = source.to_mlir()
    result = qlx.compiler.to_pbc(source)

    assert source.to_mlir() == original
    text = result.to_mlir()
    assert "cflow.repeat 1024" in text
    assert text.count("#qlx.action<pauli_rotation>") == 1
    assert "x_mask = 1 : i64" in text
    assert "z_mask = 0 : i64" in text
    assert "#qlx.action<h>" not in text
    assert "#qlx.action<t>" not in text
    assert result.module.operation.verify()


def test_pbc_materializes_a_bounded_nonidentity_repeat_phase():
    source = qlx.compiler.synthesize(
        bounded_nonidentity_residual_repeat,
        gate_set=qlx.compiler.gate_sets.clifford_t,
    )
    original = source.to_mlir()
    result = qlx.compiler.to_pbc(source)

    assert source.to_mlir() == original
    text = result.to_mlir()
    assert "cflow.repeat 1" in text
    assert text.count("#qlx.action<pauli_rotation>") == 2
    assert "x_mask = 1 : i64, z_mask = 0 : i64" in text
    assert "x_mask = 0 : i64, z_mask = 1 : i64" in text
    assert "#qlx.action<h>" not in text
    assert "#qlx.action<t>" not in text
    assert result.module.operation.verify()


def test_native_pbc_lowering_reports_failure_in_a_later_program(capfd):
    from cudaq.mlir._mlir_libs import _qlxRuntime as runtime

    module = mlir_ir.Module.parse(
        r"""
module {
  qlx.program @valid_first : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64}
        : !qlx.logical_qubit
    %r = cflow.repeat 2
        iter(%arg : !qlx.logical_qubit = %q) {
      %h = qlx.apply #qlx.action<h>(%arg)
          : (!qlx.logical_qubit) -> !qlx.logical_qubit
      %t = qlx.apply #qlx.action<t>(%h)
          : (!qlx.logical_qubit) -> !qlx.logical_qubit
      cflow.yield %t : !qlx.logical_qubit
    }
    %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
  qlx.program @invalid_second : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64}
        : !qlx.logical_qubit
    %i = qlx.apply #qlx.action<idle>(%q)
        : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %m = qlx.measure #qlx.pauli<Z> %i : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
}
""",
        mlir_ir.Context(),
    )
    with pytest.raises(RuntimeError, match="QLX PBC lowering failed"):
        runtime.lower_to_pbc_module(module)

    assert "cannot erase workload-bearing idle" in capfd.readouterr().err


def test_native_pbc_lowering_rejects_live_classical_gate_payload(capfd):
    from cudaq.mlir._mlir_libs import _qlxRuntime as runtime

    module = mlir_ir.Module.parse(
        r"""
module {
  qlx.program @valid_first : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64}
        : !qlx.logical_qubit
    %t = qlx.apply #qlx.action<t>(%q)
        : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %m = qlx.measure #qlx.pauli<Z> %t : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
  qlx.program @invalid_second : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64}
        : !qlx.logical_qubit
    %c = arith.constant 0.0 : f64
    %t:2 = qlx.apply #qlx.action<t>(%q, %c)
        : (!qlx.logical_qubit, f64) -> (!qlx.logical_qubit, i1)
    qlx.discard %t#0 : !qlx.logical_qubit
    qlx.return %t#1 : i1
  }
}
""",
        mlir_ir.Context(),
    )
    with pytest.raises(RuntimeError, match="QLX PBC lowering failed"):
        runtime.lower_to_pbc_module(module)

    assert "no classical payloads" in capfd.readouterr().err


def test_fixed_builtin_parameters_fail_all_certificates(capfd):
    from cudaq.mlir._mlir_libs import _qlxRuntime as runtime

    module = mlir_ir.Module.parse(
        r"""
module {
  qlx.program @bad_parameters : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64}
        : !qlx.logical_qubit
    %t = qlx.apply #qlx.action<t>(%q)
        : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %m = qlx.measure #qlx.pauli<Z> %t : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
}
""",
        mlir_ir.Context(),
    )
    t_apply = next(operation for operation in _walk(module.operation)
                   if operation.name == "qlx.apply")
    with module.context:
        t_apply.attributes["parameters"] = mlir_ir.Attribute.parse(
            "{unexpected = 7 : i64}", context=module.context)
    with pytest.raises(mlir_ir.MLIRError, match="parameter bindings"):
        module.operation.verify()
    assert runtime.verify_clifford_t_module(module) is False
    with pytest.raises(RuntimeError, match="QLX PBC lowering failed"):
        runtime.lower_to_pbc_module(module)

    assert "parameter bindings" in capfd.readouterr().err


def test_native_pbc_lowering_rejects_open_source_owner(capfd):
    from cudaq.mlir._mlir_libs import _qlxRuntime as runtime

    module = mlir_ir.Module.parse(
        r"""
module {
  qlx.program @bad_open_source_owner : () -> i1 attributes {qlx.stage = "p0"} {
    %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64}
        : !qlx.logical_qubit
    %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64}
        : !qlx.logical_qubit
    %t = qlx.apply #qlx.action<t>(%q0)
        : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %m = qlx.measure #qlx.pauli<Z> %t : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
}
""",
        mlir_ir.Context(),
    )
    assert module.operation.verify()
    assert runtime.verify_clifford_t_module(module) is True
    with pytest.raises(RuntimeError, match="QLX PBC lowering failed"):
        runtime.lower_to_pbc_module(module)

    assert "requires every source logical-qubit owner" in capfd.readouterr().err


def test_standalone_source_entrypoints_reject_negative_repeat_count(capfd):
    from cudaq.mlir._mlir_libs import _qlxRuntime as runtime

    module = mlir_ir.Module.parse(
        r"""
module {
  qlx.program @bad_repeat_count : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64}
        : !qlx.logical_qubit
    %r = cflow.repeat 1
        iter(%arg : !qlx.logical_qubit = %q) {
      cflow.yield %arg : !qlx.logical_qubit
    }
    %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
}
""",
        mlir_ir.Context(),
    )
    repeat = next(operation for operation in _walk(module.operation)
                  if operation.name == "cflow.repeat")
    with module.context:
        repeat.attributes["count"] = mlir_ir.Attribute.parse(
            "-1 : i64", context=module.context)
    with pytest.raises(mlir_ir.MLIRError, match="non-negative"):
        module.operation.verify()
    assert runtime.verify_clifford_t_module(module) is False
    with pytest.raises(RuntimeError, match="QLX PBC lowering failed"):
        runtime.lower_to_pbc_module(module)

    assert "non-negative" in capfd.readouterr().err


def test_standalone_pbc_certificate_rejects_negative_repeat_count():
    from cudaq.mlir._mlir_libs import _qlxRuntime as runtime

    module = mlir_ir.Module.parse(
        r"""
module {
  qlx.program @bad_repeat_count : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64}
        : !qlx.logical_qubit
    %r = cflow.repeat 1
        iter(%arg : !qlx.logical_qubit = %q) {
      cflow.yield %arg : !qlx.logical_qubit
    }
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%r)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.discard %m#0 : !qlx.logical_qubit
    qlx.return %m#1 : i1
  }
}
""",
        mlir_ir.Context(),
    )
    repeat = next(operation for operation in _walk(module.operation)
                  if operation.name == "cflow.repeat")
    with module.context:
        repeat.attributes["count"] = mlir_ir.Attribute.parse(
            "-1 : i64", context=module.context)

    with pytest.raises(mlir_ir.MLIRError, match="non-negative"):
        module.operation.verify()
    assert runtime.verify_pbc_module(module) is False


def test_pbc_preserves_authored_discard_reason_separately_from_cleanup():
    source = qlx.compiler.synthesize(
        authored_discard_reason,
        gate_set=qlx.compiler.gate_sets.clifford_t,
    )
    result = qlx.compiler.to_pbc(source)
    text = result.to_mlir()

    assert 'reason = "ancilla-cleanup"' in text
    assert text.count("qlx.discard") == 2
    assert text.index('reason = "ancilla-cleanup"') < text.rindex("qlx.discard")


def test_pbc_zero_count_retains_the_canonical_rotation_template():
    source = qlx.compiler.synthesize(
        folded_zero_count_repeat,
        gate_set=qlx.compiler.gate_sets.clifford_t,
    )
    result = qlx.compiler.to_pbc(source)
    text = result.to_mlir()

    assert source.synthesis.t_count == 1
    assert "cflow.repeat 0" in text
    assert text.count("#qlx.action<pauli_rotation>") == source.synthesis.t_count
    assert "x_mask = 1 : i64" in text
    assert "z_mask = 0 : i64" in text
