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
# Registering one with `set_compile_target` overrides what the active target's
# QPU would otherwise provide, while the launch stays where it was.

import os

import pytest

import cudaq
from cudaq._experimental import (
    CompileTarget,
    PipelineConfig,
    set_compile_target,
    set_runtime_endpoint,
)
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


def compiled_quake_ops(kernel):
    """Launch `kernel` through a capturing endpoint and return its quake ops.

    Registering the endpoint keeps the compile target that is already
    installed, so this observes the IR that the current target compiles to.
    """
    endpoint = CapturingEndpoint()
    set_runtime_endpoint(endpoint)
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
# Use with set_compile_target
# ---------------------------------------------------------------------------- #


def test_set_compile_target_rejects_non_target():
    with pytest.raises(TypeError, match="not a compile target"):
        set_compile_target("qpp-cpu")


def test_kernel_runs_with_a_custom_compile_target():
    set_compile_target(CompileTarget())

    @cudaq.kernel
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])
        mz(q)

    counts = cudaq.sample(bell, shots_count=10)
    # Bell state: only correlated outcomes should appear.
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 10


def test_pipeline_config_controls_compiled_ir():
    # Baseline: the default qpp-cpu pipeline keeps the swap intact.
    set_compile_target(CompileTarget())
    default_ops = compiled_quake_ops(make_swap_kernel())
    assert any(name == "quake.swap" for name, _ in default_ops)

    # Custom pipeline: decompose swap into CNOTs.
    cudaq.set_target("qpp-cpu")
    set_compile_target(swap_pipeline_target())
    decomposed_ops = compiled_quake_ops(make_swap_kernel())

    # The swap has been decomposed away into three controlled-x (CNOT) ops.
    assert all(name != "quake.swap" for name, _ in decomposed_ops)
    controlled_x = [
        name for name, num_operands in decomposed_ops
        if name == "quake.x" and num_operands > 1
    ]
    assert len(controlled_x) == 3


def test_pipeline_config_preserves_kernel_semantics():
    # Decomposition must not change the observable behaviour of the kernel.
    set_compile_target(CompileTarget())
    default_counts = cudaq.sample(make_swap_kernel(), shots_count=10)

    set_compile_target(swap_pipeline_target())
    decomposed_counts = cudaq.sample(make_swap_kernel(), shots_count=10)

    assert dict(decomposed_counts.items()) == dict(default_counts.items())


def test_compile_target_does_not_leak_after_switch():
    """A compile target must not survive a target change.

    Changing the target replaces the platform's QPUs, which drops the compile
    target with them. Otherwise a custom pipeline would leak into unrelated
    kernels (e.g. `cudaq.draw`).
    """

    def op_names(kernel):
        return [name for name, _ in compiled_quake_ops(kernel)]

    # Install a compile target that decomposes swaps into CNOTs.
    set_compile_target(swap_pipeline_target())
    assert "quake.swap" not in op_names(make_swap_kernel())

    # Switching targets must restore default behaviour.
    cudaq.set_target("qpp-cpu")
    assert "quake.swap" in op_names(make_swap_kernel())

    # Same expectation after a reset.
    set_compile_target(swap_pipeline_target())
    cudaq.reset_target()
    assert "quake.swap" in op_names(make_swap_kernel())


def test_support_conditionals_on_measure_results():
    ct = CompileTarget()
    ct.support_conditionals_on_measure_results = True
    set_compile_target(ct)

    @cudaq.kernel
    def kernel() -> bool:
        q = cudaq.qvector(2)
        h(q[0])
        b = mz(q[0])
        if b:
            x(q[1])
        return b == mz(q[1])

    # it runs fine
    assert kernel()

    ct.support_conditionals_on_measure_results = False
    set_compile_target(ct)

    # now it throws a runtime error
    with pytest.raises(RuntimeError):
        kernel()


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
