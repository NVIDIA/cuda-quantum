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
from cudaq._experimental import CompileTarget, CustomTarget, PipelineConfig
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


def swap_pipeline_target():
    ct = CompileTarget()
    ct.pipeline_config.override_pass_pipeline = SWAP_TO_CX_PIPELINE
    return ct


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


def test_pipeline_config_controls_compiled_ir():
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


def test_pipeline_config_equality_and_hash():
    a = PipelineConfig()
    b = PipelineConfig()
    assert a == b
    assert hash(a) == hash(b)

    b.disable_qubit_mapping = True
    assert a != b


def test_compile_target_equality_and_hash():
    a = CompileTarget()
    b = CompileTarget()
    assert a == b
    assert hash(a) == hash(b)

    b.pipeline_config.override_pass_pipeline = SWAP_TO_CX_PIPELINE
    assert a != b
    assert hash(a) != hash(b)


def test_repr_is_informative():
    assert "PipelineConfig(" in repr(PipelineConfig())
    assert "CompileTarget(" in repr(CompileTarget())
    assert SWAP_TO_CX_PIPELINE in repr(swap_pipeline_target())


if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
