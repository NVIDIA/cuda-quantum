# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The `qpu_layout` target: a compile target plus a runtime endpoint.

The compile target lowers a kernel to virtual-qubit wire-set Quake and stops
there -- no decomposition and no qubit mapping, because the layout model makes
its own placement and mapping decisions. The endpoint receives that module and
traces it instead of simulating quantum state.

    import cudaq
    from cudaq_qpu_layout import QpuLayoutTarget

    target = QpuLayoutTarget.build(num_regions=2, region_size=2)
    cudaq.set_target(target)
    cudaq.sample(my_kernel)
    trace = target.runtime_endpoint.trace

Counts come back all-zero: nothing here evaluates a quantum state. The trace is
the result.
"""

from dataclasses import dataclass, field

import cudaq
from cudaq._experimental import CompileTarget, CustomTarget
from cudaq._experimental.runtime_endpoint import RuntimeEndpoint
from cudaq.mlir.passmanager import PassManager

from .model import QpuModel
from .sim import simulate_module

# Lower to virtual-qubit wire-set Quake and stop. `prepare-for-wireset` adds the
# wire set and assigns wire indices in one pass; no decomposition runs, since the
# model does not care about a gate set, and no mapping runs, since placement is
# the thing being modeled.
WIRESET_PIPELINE = "prepare-for-wireset{add-wireset=true}"

# `nop` codegen hands the endpoint Quake rather than QIR -- and, as a side
# effect, forces `no-loop-unroll=true`, so structured control flow survives into
# the module. The layout model schedules straight-line code, so unroll here and
# stop short of `lower-to-cfg`, which would split the body into blocks.
UNROLL_PIPELINE = (
    "builtin.module("
    "canonicalize,distributed-device-call,cse,"
    "func.func("
    "memtoreg,canonicalize,cc-loop-normalize,"
    "cc-loop-unroll{maximum-iterations=1024 "
    "signal-failure-if-any-loop-cannot-be-completely-unrolled=true "
    "allow-early-exit=true},"
    "canonicalize"
    "),"
    "canonicalize,cse,symbol-dce"
    ")")


def build_compile_target():
    """The compile half: wire-set Quake, delivered as Quake."""
    target = CompileTarget()
    target.pipeline_config.mid_level_pipeline = WIRESET_PIPELINE
    target.pipeline_config.low_level_pipeline = ""
    target.pipeline_config.codegen_translation = "nop"
    target.pipeline_config.disable_qubit_mapping = True
    return target


class QpuLayoutEndpoint(RuntimeEndpoint):
    """The launch half: lays the module out on the QPU and records a trace.

    The most recent trace is kept on `trace` (and `builder`, for callers that
    want to render a viewer from it).
    """

    def __init__(self, model=None):
        self.model = model or QpuModel()
        self.builder = None
        self.trace = None

    def sample(self, module, args, **options):
        self.builder = self._run(module)
        self.trace = self.builder.to_json()
        shots = options.get("shots_count", 0)
        # Nothing here evaluates a state, so every shot reads back zero. The
        # width is the circuit's, so the result still shapes like the kernel.
        width = self.trace["summary"]["num_vqubits"]
        return cudaq.SampleResult({"0" * width: shots}) if width else \
            cudaq.SampleResult({})

    def _run(self, module):
        mlir = module.mlir_module
        pm = PassManager.parse(UNROLL_PIPELINE, context=mlir.context)
        try:
            pm.run(mlir.operation)
        except Exception as e:
            raise RuntimeError(
                f"Failed to unroll the layout payload: {e}\n{mlir}") from e
        return simulate_module(mlir, self.model)


@dataclass
class QpuLayoutTarget(CustomTarget):
    """Pairs the two halves so `cudaq.set_target` installs them together."""

    runtime_endpoint: RuntimeEndpoint = field(
        default_factory=QpuLayoutEndpoint)
    compile_target: CompileTarget = field(default_factory=build_compile_target)

    @classmethod
    def build(cls, **model_kwargs):
        """A target whose endpoint models a QPU described by `model_kwargs`."""
        return cls(runtime_endpoint=QpuLayoutEndpoint(QpuModel(**model_kwargs)))
