# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from contextlib import contextmanager

import cudaq
from cudaq._experimental import (
    CompileTarget,
    CustomPipelineLowering,
    CustomTarget,
    QuakeReferenceSemanticsProfile,
)

basis = ["h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)"]

# Define a custom target, with the pipeline inferred by the compiler.
inferred = CompileTarget(
    ir_output_profile=QuakeReferenceSemanticsProfile(
        basis_gates=basis, measurement_type="single_qubit"),
    codegen="qir-base",
)
# The target has one compilation stage, with implicit lowering.
assert len(inferred.stages) == 1
assert inferred.stages[0].lowering is None

# Alternatively, a target can also be defined by specifying the compilation
# pipeline explicitly.
explicit = CompileTarget(
    custom_pipeline=CustomPipelineLowering(
        high_level="expand-measurements",
        mid_level="basis-conversion{basis=" + ",".join(basis) + "}",
    ),
    codegen="qir-base",
)
# The target has one compilation stage, with an explicit custom lowering.
assert len(explicit.stages) == 1
assert explicit.stages[0].lowering is not None


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


@contextmanager
def set_target_context(*args, **kwargs):
    """Set the target context for the duration of the context manager."""
    cudaq.set_target(*args, **kwargs)
    try:
        yield
    finally:
        cudaq.reset_target()


# Run the kernel on an existing target, using the specified compile pipeline.
with set_target_context("qpp-cpu", compile_target=explicit):
    res = cudaq.sample(bell, shots_count=1)
    assert "00" in res or "11" in res


# Or alternatively, define your custom runtime endpoint:
class InspectEndpoint:
    """Captures the lowered MLIR instead of executing the kernel."""

    def sample(self, module, args, **kwargs):
        self.mlir = str(module.mlir_module)
        return cudaq.SampleResult()


# and customize both the compile target and runtime endpoint:
endpoint = InspectEndpoint()
target = CustomTarget(compile_target=inferred, runtime_endpoint=endpoint)
with set_target_context(target):
    res = cudaq.sample(bell, shots_count=1)
    assert "quake.mz" in endpoint.mlir
    print(endpoint.mlir)
