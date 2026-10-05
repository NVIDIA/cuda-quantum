# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for the experimental `CustomTarget` type and the `cudaq.set_target`
# overload that installs a compile target and runtime endpoint together.

import pytest
from dataclasses import dataclass, field

import cudaq
from cudaq._experimental import CompileTarget, CustomTarget
from cudaq._experimental.runtime_endpoint import RuntimeEndpoint
from cudaq.mlir.ir import WalkResult

SWAP_TO_CX_PIPELINE = (
    "canonicalize,decomposition{enable-patterns=SwapToCX},canonicalize")


@pytest.fixture(autouse=True)
def reset_target():
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


@cudaq.kernel
def swap_kernel():
    q = cudaq.qvector(2)
    x(q[0])
    swap(q[0], q[1])
    mz(q)


class DemoEndpoint(RuntimeEndpoint):
    """Records launches and returns a canned result."""

    def __init__(self):
        self.calls = []
        self.mlir_module = None

    def sample(self, module, args, **kwargs):
        self.calls.append(("sample", kwargs))
        self.mlir_module = module.mlir_module
        return cudaq.SampleResult({"00": kwargs["shots_count"]})


@dataclass
class DemoCustomTarget(CustomTarget):
    runtime_endpoint: RuntimeEndpoint = field(default_factory=DemoEndpoint)
    compile_target: CompileTarget = field(default_factory=CompileTarget)


def swap_pipeline_target():
    ct = CompileTarget()
    ct.pipeline_config.override_pass_pipeline = SWAP_TO_CX_PIPELINE
    return ct


def compiled_quake_ops(kernel, target):
    ops = []

    def walk(op):
        ops.append((op.name, len(op.operands)))
        return WalkResult.ADVANCE

    cudaq.set_target(target)
    cudaq.sample(kernel, shots_count=1)
    endpoint = target.runtime_endpoint
    assert endpoint.mlir_module is not None
    endpoint.mlir_module.operation.walk(walk)
    return ops


def test_set_target_accepts_custom_target_subclass():
    target = DemoCustomTarget()
    cudaq.set_target(target)

    result = cudaq.sample(swap_kernel, shots_count=7)

    assert result["00"] == 7
    assert len(target.runtime_endpoint.calls) == 1
    assert target.runtime_endpoint.calls[0] == ("sample", {
        "shots_count": 7,
        "explicit_measurements": False
    })


def test_set_target_accepts_direct_custom_target_instance():
    endpoint = DemoEndpoint()
    target = CustomTarget(
        runtime_endpoint=endpoint,
        compile_target=CompileTarget(),
    )
    cudaq.set_target(target)

    cudaq.sample(swap_kernel, shots_count=3)

    assert len(endpoint.calls) == 1


def test_set_target_with_custom_target_uses_custom_compile_target():
    target = DemoCustomTarget(compile_target=swap_pipeline_target())
    ops = compiled_quake_ops(swap_kernel, target)

    assert all(name != "quake.swap" for name, _ in ops)
    controlled_x = [
        name for name, num_operands in ops
        if name == "quake.x" and num_operands > 1
    ]
    assert len(controlled_x) == 3


def test_duck_typed_object_is_not_accepted_as_custom_target():

    class DuckTarget:
        runtime_endpoint = DemoEndpoint()
        compile_target = CompileTarget()

    with pytest.raises(TypeError):
        cudaq.set_target(DuckTarget())


def test_set_target_rejects_kwargs_for_custom_target():
    target = DemoCustomTarget()

    with pytest.raises(
            TypeError,
            match="does not accept keyword arguments when target is a "
            "cudaq._experimental.CustomTarget",
    ):
        cudaq.set_target(target, option="fp64")


def test_named_set_target_still_works_after_wrapper():
    cudaq.set_target("qpp-cpu")
    counts = cudaq.sample(swap_kernel, shots_count=10)
    assert sum(counts.values()) == 10
