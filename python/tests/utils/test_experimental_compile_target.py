# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
Tests for the experimental ``CompileTarget`` bindings and their use with `cudaq.set_target`.
"""

import os

import pytest

import cudaq
from cudaq._experimental.target import (
    CompileTarget,
    PipelineConfig,
    RuntimeEndpoint,
)
from cudaq.mlir.ir import WalkResult


@pytest.fixture(autouse=True)
def reset_target():
    """Ensure every test starts and ends from a well-defined target."""
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


def test_runtime_endpoint():
    re = RuntimeEndpoint("qpp-cpu")
    assert re.name == "qpp-cpu"
    assert re.options == {}

    re = RuntimeEndpoint("quantinuum", {"machine": "H1-1", "emulate": True})
    assert re.name == "quantinuum"
    assert re.options == {"machine": "H1-1", "emulate": "true"}

    re.name = "stim"
    re.options = {"shots": 100}
    assert re.name == "stim"
    assert re.options == {"shots": "100"}


# ---------------------------------------------------------------------------- #
# Use with cudaq.set_target
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["qpp-cpu", "stim"])
def test_set_target_with_compile_target_runs_kernel(name):
    ct = CompileTarget(runtime_endpoint=name)
    cudaq.set_target(ct)

    assert cudaq.get_target().name == name

    counts = cudaq.sample(bell, shots_count=10)
    # Bell state: only correlated outcomes should appear.
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 10


def test_set_target_switches_runtime_endpoint():
    cudaq.set_target(CompileTarget(runtime_endpoint="qpp-cpu"))
    assert cudaq.get_target().name == "qpp-cpu"

    cudaq.set_target(CompileTarget(runtime_endpoint="stim"))
    assert cudaq.get_target().name == "stim"


def test_set_target_with_empty_runtime_endpoint_raises():
    ct = CompileTarget(runtime_endpoint="")
    with pytest.raises(RuntimeError):
        cudaq.set_target(ct)


def test_set_target_with_unknown_runtime_endpoint_raises():
    ct = CompileTarget(runtime_endpoint="not-a-real-target")
    with pytest.raises(RuntimeError):
        cudaq.set_target(ct)


def test_pipeline_config_controls_compiled_ir():
    # TODO: This test has to launch the kernel and then dig the lowered module out
    # of `cachedCompiledModule().mlir_module`. A compilation endpoint would simplify this.

    def make_swap_kernel():
        # Build a fresh kernel per call so each has its own compiled-module
        # cache slot and cannot reuse a previously compiled artifact.
        @cudaq.kernel
        def swap_kernel():
            q = cudaq.qvector(2)
            x(q[0])
            swap(q[0], q[1])
            mz(q)

        return swap_kernel

    def quake_ops(kernel):
        module = kernel.cachedCompiledModule().mlir_module
        assert module is not None, "no compiled module was cached after launch"

        def visit(op):
            if op.name.startswith("quake."):
                ops.append((op.name, len(op.operands)))
            return WalkResult.ADVANCE

        ops = []
        module.operation.walk(visit)
        return ops

    # Baseline: the default qpp-cpu pipeline keeps the swap intact.
    cudaq.set_target(CompileTarget(runtime_endpoint="qpp-cpu"))
    default_kernel = make_swap_kernel()
    default_counts = cudaq.sample(default_kernel, shots_count=1)
    default_ops = quake_ops(default_kernel)
    assert any(name == "quake.swap" for name, _ in default_ops)

    # Custom pipeline: decompose swap into CNOTs.
    ct = CompileTarget(runtime_endpoint="qpp-cpu")
    ct.pipeline_config.override_pass_pipeline = (
        "canonicalize,decomposition{enable-patterns=SwapToCX},canonicalize")
    cudaq.set_target(ct)

    decomposed_kernel = make_swap_kernel()
    decomposed_counts = cudaq.sample(decomposed_kernel, shots_count=1)
    decomposed_ops = quake_ops(decomposed_kernel)

    # The swap has been decomposed away into three controlled-x (CNOT) ops.
    assert all(name != "quake.swap" for name, _ in decomposed_ops)
    controlled_x = [
        name for name, num_operands in decomposed_ops
        if name == "quake.x" and num_operands > 1
    ]
    assert len(controlled_x) == 3

    # Decomposition must not change the observable behaviour of the kernel.
    assert dict(decomposed_counts.items()) == dict(default_counts.items())


def test_support_conditionals_on_measure_results():
    ct = CompileTarget(runtime_endpoint="qpp-cpu")
    ct.support_conditionals_on_measure_results = True
    cudaq.set_target(ct)

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
    cudaq.set_target(ct)

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


def test_runtime_endpoint_equality_and_hash():
    a = RuntimeEndpoint("qpp-cpu", {"k": "v"})
    b = RuntimeEndpoint("qpp-cpu", {"k": "v"})
    assert a == b
    assert hash(a) == hash(b)

    assert RuntimeEndpoint("qpp-cpu") != RuntimeEndpoint("stim")
    assert RuntimeEndpoint("qpp-cpu", {"k": "v"}) != RuntimeEndpoint(
        "qpp-cpu", {"k": "w"})


def test_compile_target_equality_and_hash():
    a = CompileTarget()
    b = CompileTarget()
    assert a == b
    assert hash(a) == hash(b)

    b.runtime_endpoint = "stim"
    assert a != b


def test_repr_is_informative():
    assert "PipelineConfig(" in repr(PipelineConfig())
    assert "RuntimeEndpoint(name='qpp-cpu'" in repr(RuntimeEndpoint("qpp-cpu"))
    assert "CompileTarget(" in repr(CompileTarget())


if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
