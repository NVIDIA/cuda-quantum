# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Lay a kernel out on a region-based QPU and record what it costs.

Two ways in. Through a CUDA-Q target:

    import cudaq
    from cudaq_qpu_layout import QpuLayoutTarget

    target = QpuLayoutTarget.build(num_regions=2, region_size=2)
    cudaq.set_target(target)
    cudaq.sample(my_kernel)
    trace = target.runtime_endpoint.trace

Or directly on a Quake payload, with no target and no compilation:

    from cudaq_qpu_layout import QpuModel, simulate
    trace = simulate(open("payload.mlir").read(), QpuModel()).to_json()
"""

from .model import QpuModel
from .sim import LayoutError, simulate, simulate_module
from .target import QpuLayoutEndpoint, QpuLayoutTarget, build_compile_target
from .trace import replay, summarize
from .viewer import write_viewer

__all__ = [
    "LayoutError",
    "QpuLayoutEndpoint",
    "QpuLayoutTarget",
    "QpuModel",
    "build_compile_target",
    "replay",
    "simulate",
    "simulate_module",
    "summarize",
    "write_viewer",
]
