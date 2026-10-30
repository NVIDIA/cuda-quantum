# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for `cudaq.compile`, which compiles a kernel for a target without
# launching it.

import pytest

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


@cudaq.kernel
def rotation_kernel(theta: float):
    q = cudaq.qubit()
    rx(theta, q)
    mz(q)


class RecordingEndpoint(RuntimeEndpoint):

    def __init__(self, supports_jit=True):
        self.supports_jit = supports_jit
        self.calls = []

    def sample(self, module, args, **kwargs):
        self.calls.append(kwargs)
        return cudaq.SampleResult({})


def op_names(module):
    names = []

    def walk(op):
        names.append(op.name)
        return WalkResult.ADVANCE

    module.operation.walk(walk)
    return names


def test_compile_for_the_current_platform():
    compiled = cudaq.compile(swap_kernel)

    assert compiled.mlir_module is not None
    # A local simulator launch JIT-compiles the kernel.
    assert compiled.entry_point is not None
    assert "quake.swap" in op_names(compiled.mlir_module)


def test_compile_with_arguments():
    compiled = cudaq.compile(rotation_kernel, 0.5)

    assert compiled.mlir_module is not None


def test_compile_uses_the_pipeline_of_the_given_target():
    endpoint = RecordingEndpoint()
    target = CustomTarget(
        runtime_endpoint=endpoint,
        compile_target=CompileTarget(lowering=SWAP_TO_CX_PIPELINE),
    )

    compiled = cudaq.compile(swap_kernel, target=target)

    assert "quake.swap" not in op_names(compiled.mlir_module)
    # The installed target is unchanged and the kernel was not launched.
    assert "quake.swap" in op_names(cudaq.compile(swap_kernel).mlir_module)
    assert endpoint.calls == []


@pytest.mark.parametrize("supports_jit", [True, False])
def test_compile_builds_a_jit_entry_point_only_if_the_endpoint_consumes_it(
        supports_jit):
    target = CustomTarget(
        runtime_endpoint=RecordingEndpoint(supports_jit=supports_jit),
        compile_target=CompileTarget(),
    )

    compiled = cudaq.compile(swap_kernel, target=target)

    assert compiled.mlir_module is not None
    assert (compiled.entry_point is not None) == supports_jit


def test_compile_does_not_install_the_target():
    target = CustomTarget(runtime_endpoint=RecordingEndpoint(),
                          compile_target=CompileTarget())

    cudaq.compile(swap_kernel, target=target)

    assert cudaq.get_target().name == "qpp-cpu"


def test_compile_rejects_a_bare_compile_target():
    with pytest.raises(TypeError, match="CustomTarget"):
        cudaq.compile(swap_kernel, target=CompileTarget())
