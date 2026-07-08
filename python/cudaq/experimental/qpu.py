# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Protocols and types for QPU definitions in Python."""

from typing import Protocol, runtime_checkable

from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import (
    CompiledModule,
    PipelineConfig,
    CompileTarget,
    KernelArgs,
    ObserveResult,
    SampleResult,
)


@runtime_checkable
class SupportsSampleQPU(Protocol):
    """Protocol for QPUs that support ``cudaq.sample``."""

    def get_compile_target_sample(self) -> CompileTarget:
        """Return compilation settings for a sample launch."""
        ...

    def launch_sample(
        self,
        module: CompiledModule,
        args: KernelArgs,
    ) -> SampleResult:
        """Execute a compiled kernel under the sample policy."""
        ...


@runtime_checkable
class SupportsObserveQPU(Protocol):
    """Protocol for QPUs that support ``cudaq.observe``."""

    def get_compile_target_observe(self) -> CompileTarget:
        """Return compilation settings for an observe launch."""
        ...

    def launch_observe(
        self,
        module: CompiledModule,
        args: KernelArgs,
    ) -> ObserveResult:
        """Execute a compiled kernel under the observe policy."""
        ...
