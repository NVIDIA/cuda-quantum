# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for the experimental `CompileTarget` bindings.
#
# A compile target owns the compilation half of a backend: the pass pipelines,
# the code generation and the capabilities kernels are compiled against.
# Install it together with a runtime endpoint as a `CustomTarget`.

import os

import pytest

import cudaq
from cudaq._experimental import (Codegen, CompileTarget, CustomPipelineLowering,
                                 CustomTarget)
from cudaq._target import CompilationStage
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.ir import WalkResult

# Decompose swaps into CNOTs instead of running the default pipeline.
SWAP_TO_CX_PIPELINE = (
    "canonicalize,decomposition{enable-patterns=SwapToCX},canonicalize")


@pytest.fixture(autouse=True)
def reset_target():
    """Ensure every test starts and ends from a well-defined target."""
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def make_swap_kernel():
    # Build a fresh kernel per call so each has its own compiled-module cache
    # slot and cannot reuse a previously compiled artifact.

    @cudaq.kernel
    def swap_kernel():
        q = cudaq.qvector(2)
        x(q[0])
        swap(q[0], q[1])
        mz(q)

    return swap_kernel


def test_frontend_reexports_shared_bindings():
    from importlib import import_module
    from types import ModuleType
    from cudaq.core import backends
    from cudaq.mlir._mlir_libs import _backends, _quakeDialects, _quakeDialectsCore

    assert Codegen is backends.Codegen
    assert CompileTarget is backends.CompileTarget
    assert CustomPipelineLowering is backends.CustomPipelineLowering
    assert CustomTarget is backends.CustomTarget
    assert cudaq.Resources is backends.Resources
    assert cudaq.EstimateResult is backends.EstimateResult
    # Every public native export must retain object identity through the
    # frontend, including imports using its original submodule paths.
    for source, destination in ((_quakeDialectsCore, _quakeDialects),
                                (_backends, cudaq_runtime)):
        for name, value in vars(source).items():
            if name.startswith("_"):
                continue
            assert getattr(destination, name) is value
            if isinstance(value, ModuleType):
                assert import_module(f"{destination.__name__}.{name}") is value


def test_target_values_have_one_registration():
    from cudaq import _experimental, _target
    from cudaq.core import backends
    from cudaq.mlir._mlir_libs import _backends

    for name in ("Codegen", "CompilationStage", "CompileTarget",
                 "CustomPipelineLowering", "QuakeReferenceSemanticsProfile",
                 "UncheckedProfile"):
        value = getattr(_target, name)
        for module in (_experimental, backends, _backends, cudaq_runtime):
            assert getattr(module, name) is value


def test_custom_target_runtime_annotations():
    from dataclasses import fields
    from typing import get_type_hints
    from cudaq._experimental import RuntimeEndpoint
    from cudaq.core import backends

    expected = {
        "runtime_endpoint": RuntimeEndpoint,
        "compile_target": CompileTarget,
    }
    assert get_type_hints(CustomTarget) == expected
    assert {
        field.name: field.type for field in fields(CustomTarget)
    } == expected
    assert RuntimeEndpoint is backends.RuntimeEndpoint


def swap_pipeline_target():
    return CompileTarget(custom_pipeline=SWAP_TO_CX_PIPELINE)


class CapturingEndpoint:
    """A runtime endpoint that keeps the MLIR it was handed."""

    def __init__(self):
        self.mlir_module = None

    def sample(self, module, args, **kwargs):
        self.mlir_module = module.mlir_module
        return cudaq.SampleResult()


def set_custom_target(compile_target, endpoint=None):
    if endpoint is None:
        endpoint = CapturingEndpoint()
    cudaq.set_target(
        CustomTarget(runtime_endpoint=endpoint, compile_target=compile_target))


def compiled_quake_ops(kernel, compile_target=None):
    """Launch `kernel` through a capturing endpoint and return its quake ops."""
    if compile_target is None:
        compile_target = CompileTarget()
    endpoint = CapturingEndpoint()
    set_custom_target(compile_target, endpoint=endpoint)
    cudaq.sample(kernel, shots_count=1)

    module = endpoint.mlir_module
    assert module is not None, "the endpoint was not handed a compiled module"

    ops = []

    def visit(op):
        if op.name.startswith("quake."):
            ops.append((op.name, len(op.operands)))
        return WalkResult.ADVANCE

    module.operation.walk(visit)
    return ops


def test_lowering_controls_compiled_ir():
    # Baseline: the default qpp-cpu pipeline keeps the swap intact.
    default_ops = compiled_quake_ops(make_swap_kernel())
    assert any(name == "quake.swap" for name, _ in default_ops)

    # Custom pipeline: decompose swap into CNOTs.
    decomposed_ops = compiled_quake_ops(make_swap_kernel(),
                                        compile_target=swap_pipeline_target())

    # The swap has been decomposed away into three controlled-x (CNOT) ops.
    assert all(name != "quake.swap" for name, _ in decomposed_ops)
    controlled_x = [
        name for name, num_operands in decomposed_ops
        if name == "quake.x" and num_operands > 1
    ]
    assert len(controlled_x) == 3


def test_compile_target_does_not_leak_after_switch():
    """A compile target must not survive a target change.

    Changing the target replaces the platform's QPUs, which drops the compile
    target with them. Otherwise a custom pipeline would leak into unrelated
    kernels (e.g. `cudaq.draw`).
    """

    def op_names(kernel, compile_target=None):
        return [name for name, _ in compiled_quake_ops(kernel, compile_target)]

    # Install a compile target that decomposes swaps into CNOTs.
    assert "quake.swap" not in op_names(make_swap_kernel(),
                                        compile_target=swap_pipeline_target())

    # Switching targets must restore default behaviour.
    cudaq.set_target("qpp-cpu")
    assert "quake.swap" in op_names(make_swap_kernel())

    # Same expectation after a reset.
    set_custom_target(swap_pipeline_target())
    cudaq.reset_target()
    assert "quake.swap" in op_names(make_swap_kernel())


def test_support_explicit_measurements():
    ct = CompileTarget()
    ct.support_explicit_measurements = False
    set_custom_target(ct)
    assert cudaq_runtime.supportsExplicitMeasurements() is False

    ct.support_explicit_measurements = True
    set_custom_target(ct)
    assert cudaq_runtime.supportsExplicitMeasurements() is True


# ---------------------------------------------------------------------------- #
# Use with a named target
# ---------------------------------------------------------------------------- #


def test_named_target_with_compile_target():
    ct = CompileTarget()
    ct.support_explicit_measurements = False
    cudaq.set_target("qpp-cpu", compile_target=ct)
    assert cudaq.get_target().name == "qpp-cpu"
    assert cudaq_runtime.supportsExplicitMeasurements() is False

    # A new seed must not rebuild the compile target from the target config.
    cudaq.set_random_seed(13)
    assert cudaq_runtime.supportsExplicitMeasurements() is False

    # Without compile_target, the target's default compile target is used.
    cudaq.set_target("qpp-cpu")
    assert cudaq_runtime.supportsExplicitMeasurements() is True


def test_named_target_compiles_against_given_compile_target():
    cudaq.set_target(
        "qpp-cpu",
        compile_target=CompileTarget(custom_pipeline="not-a-registered-pass"))
    with pytest.raises(RuntimeError, match="not-a-registered-pass"):
        cudaq.sample(make_swap_kernel(), shots_count=1)

    cudaq.set_target(cudaq.get_target("qpp-cpu"),
                     compile_target=swap_pipeline_target())
    assert cudaq.sample(make_swap_kernel(), shots_count=10)["01"] == 10


# ---------------------------------------------------------------------------- #
# Equality / hashing / repr
# ---------------------------------------------------------------------------- #


def test_custom_pipeline_lowering_equality_and_hash():
    a = CustomPipelineLowering()
    b = CustomPipelineLowering()
    assert a == b
    assert hash(a) == hash(b)

    b.high_level = SWAP_TO_CX_PIPELINE
    assert a != b


def test_compile_target_equality_and_hash():
    a = CompileTarget()
    b = CompileTarget()
    assert a == b
    assert hash(a) == hash(b)

    b.stages = swap_pipeline_target().stages
    assert a != b
    assert hash(a) != hash(b)


def test_reading_stages_does_not_mutate_target():
    a = CompileTarget()
    b = CompileTarget()
    assert a.stages == []
    assert a == b
    assert hash(a) == hash(b)


def test_empty_stages_roundtrip():
    ct = swap_pipeline_target()
    assert ct != CompileTarget()
    ct.stages = []
    assert ct == CompileTarget()
    assert ct.stages == []


def test_staged_lowering():
    ct = CompileTarget(custom_pipeline=SWAP_TO_CX_PIPELINE)
    assert ct.stages[0].lowering.high_level == SWAP_TO_CX_PIPELINE
    assert ct.stages[0].lowering.override_pass_pipeline == ""

    # The high-level pipeline runs around the built-in passes: the swap is
    # still decomposed into three controlled-x (CNOT) ops.
    decomposed_ops = compiled_quake_ops(make_swap_kernel(), compile_target=ct)
    assert all(name != "quake.swap" for name, _ in decomposed_ops)
    controlled_x = [
        name for name, num_operands in decomposed_ops
        if name == "quake.x" and num_operands > 1
    ]
    assert len(controlled_x) == 3


def test_codegen_defaults_and_equality():
    a = Codegen()
    b = Codegen()
    assert a == b
    assert hash(a) == hash(b)
    assert a.emission_format == ""
    assert a.post_codegen_passes == ""

    b.post_codegen_passes = "canonicalize"
    assert a != b


def test_string_codegen_matches_codegen_object():
    via_string = CompileTarget(codegen="qir-base")
    via_codegen = CompileTarget(codegen=Codegen("qir-base"))
    assert via_string == via_codegen
    assert hash(via_string) == hash(via_codegen)

    a = CompileTarget()
    b = CompileTarget()
    a.codegen = "qir-base"
    b.codegen = Codegen("qir-base")
    assert a == b
    assert a.codegen.emission_format == "qir-base"


def test_codegen_in_place_mutation_updates_target():
    ct = CompileTarget(
        codegen=Codegen("qir-base", post_codegen_passes="canonicalize"))
    ct.codegen.post_codegen_passes = "remove-measurements"
    assert ct.codegen.post_codegen_passes == "remove-measurements"
    assert ct.codegen.emission_format == "qir-base"

    a = CompileTarget()
    a.codegen.emission_format = "qir-base"
    assert a.codegen == Codegen("qir-base")


def test_override_and_staged_pipelines_are_mutually_exclusive():
    with pytest.raises(ValueError):
        CustomPipelineLowering(SWAP_TO_CX_PIPELINE, high_level="canonicalize")


def test_repr_is_informative():
    assert "CustomPipelineLowering" in repr(CustomPipelineLowering())
    assert SWAP_TO_CX_PIPELINE in repr(
        CustomPipelineLowering(SWAP_TO_CX_PIPELINE))
    assert "qir:" in repr(Codegen("qir:"))
    assert "canonicalize" in repr(Codegen("qir-base", "canonicalize"))
    assert "stages=[]" in repr(CompileTarget())
    assert "custom-pipeline" in repr(swap_pipeline_target())
    assert SWAP_TO_CX_PIPELINE in repr(swap_pipeline_target())


def make_static_bell_kernel():

    @cudaq.kernel
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])
        mz(q)

    return bell


def compile_profile_target(kernel, controlled_gate):
    from cudaq._experimental import QuakeReferenceSemanticsProfile

    profile = QuakeReferenceSemanticsProfile(
        basis_gates=["h", "t", controlled_gate],
        measurement_type="single_qubit")
    target = CompileTarget(ir_output_profile=profile, codegen="qir-base")
    endpoint = CapturingEndpoint()
    set_custom_target(target, endpoint)
    cudaq.sample(kernel, shots_count=1)
    assert endpoint.mlir_module is not None
    ir = str(endpoint.mlir_module)
    assert ir.count("= quake.mz ") == 2
    assert not any(
        "quake.mz" in line and "!quake.veq" in line for line in ir.splitlines())
    assert "quake." + controlled_gate[0] + " [" in ir
    return ir


def test_profile_target_switching_uses_the_same_kernel_cache_safely():
    kernel = make_static_bell_kernel()
    z_ir = compile_profile_target(kernel, "z(1)")
    x_ir = compile_profile_target(kernel, "x(1)")
    assert "quake.x [" not in z_ir
    assert "quake.z [" not in x_ir
    assert compile_profile_target(kernel, "z(1)") == z_ir


def test_multiple_stages_are_rejected():
    from cudaq._experimental import QuakeReferenceSemanticsProfile

    stages = [
        CompilationStage(
            QuakeReferenceSemanticsProfile(basis_gates=["h", "t", gate],
                                           measurement_type="single_qubit"))
        for gate in ("z(1)", "x(1)")
    ]
    set_custom_target(CompileTarget(stages=stages, codegen="qir-base"))
    with pytest.raises(RuntimeError,
                       match="Multi-stage compilation is currently not"):
        cudaq.sample(make_static_bell_kernel(), shots_count=1)


if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
