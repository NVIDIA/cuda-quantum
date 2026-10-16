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
    return CompileTarget(lowering=SWAP_TO_CX_PIPELINE)


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


# ---------------------------------------------------------------------------- #
# Use with CustomTarget
# ---------------------------------------------------------------------------- #


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

    b.lowering = SWAP_TO_CX_PIPELINE
    assert a != b
    assert hash(a) != hash(b)


def test_reading_lowering_does_not_mutate_target():
    a = CompileTarget()
    b = CompileTarget()
    assert a.lowering is None
    assert a == b
    assert hash(a) == hash(b)


def test_lowering_none_roundtrip():
    ct = CompileTarget(lowering=SWAP_TO_CX_PIPELINE)
    assert ct != CompileTarget()
    ct.lowering = None
    assert ct == CompileTarget()
    assert ct.lowering is None


def test_string_lowering_matches_custom_pipeline_lowering():
    via_string = CompileTarget(lowering=SWAP_TO_CX_PIPELINE)
    via_lowering = CompileTarget(
        lowering=CustomPipelineLowering(SWAP_TO_CX_PIPELINE))
    assert via_string == via_lowering
    assert hash(via_string) == hash(via_lowering)

    a = CompileTarget()
    b = CompileTarget()
    a.lowering = SWAP_TO_CX_PIPELINE
    b.lowering = CustomPipelineLowering(SWAP_TO_CX_PIPELINE)
    assert a == b
    assert a.lowering.override_pass_pipeline == SWAP_TO_CX_PIPELINE


def test_staged_lowering():
    ct = CompileTarget(lowering=CustomPipelineLowering(
        high_level=SWAP_TO_CX_PIPELINE))
    assert ct.lowering.high_level == SWAP_TO_CX_PIPELINE
    assert ct.lowering.override_pass_pipeline == ""

    # The high-level pipeline runs around the built-in passes: the swap is
    # still decomposed into three controlled-x (CNOT) ops.
    decomposed_ops = compiled_quake_ops(make_swap_kernel(), compile_target=ct)
    assert all(name != "quake.swap" for name, _ in decomposed_ops)
    controlled_x = [
        name for name, num_operands in decomposed_ops
        if name == "quake.x" and num_operands > 1
    ]
    assert len(controlled_x) == 3


def test_lowering_in_place_mutation_updates_target():
    ct = CompileTarget(lowering="canonicalize")
    ct.lowering.high_level = SWAP_TO_CX_PIPELINE
    assert ct.lowering.high_level == SWAP_TO_CX_PIPELINE
    assert ct.lowering.override_pass_pipeline == "canonicalize"


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
    assert repr(CustomPipelineLowering()
               ) == "CustomPipelineLowering(high_level='', mid_level='', low_level='')"
    assert repr(CustomPipelineLowering(SWAP_TO_CX_PIPELINE)
               ) == f"CustomPipelineLowering(pipeline='{SWAP_TO_CX_PIPELINE}')"
    assert repr(Codegen("qir:")) == "Codegen('qir:')"
    assert repr(Codegen("qir-base", post_codegen_passes="canonicalize")) == (
        "Codegen('qir-base', post_codegen_passes='canonicalize')")
    assert repr(CompileTarget()
               ) == "CompileTarget(lowering=None, codegen=Codegen('qir:'))"
    assert repr(swap_pipeline_target()) == (
        "CompileTarget(lowering="
        f"CustomPipelineLowering(pipeline='{SWAP_TO_CX_PIPELINE}'),"
        " codegen=Codegen('qir:'))")


if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
