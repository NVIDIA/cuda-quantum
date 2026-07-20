# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Internal compiler-facing tooling for CUDA-Q.

This package is the Python control plane for the Optimization Validation Core:
an independent `validator` that decides whether a candidate compiler pass/pipeline
preserves circuit semantics and how it moves declared metrics. It is a
verification tool only. It never orchestrates optimization campaigns, spawns
agents, edits sources, or selects a winning revision.

The leading underscore marks this package as internal: its request/result
contracts are not yet a supported public API. Invoke the CLI from a standalone
CUDA-Q checkout with ``python3 -m cudaq._compiler.optimization_cli``.
"""

from .optimization_validation import (
    CaseResult,
    MetricDelta,
    MetricSpec,
    OracleSpec,
    PipelineSpec,
    ValidationCapabilities,
    ValidationRequest,
    ValidationResult,
    ValidationStatus,
    capabilities,
    result_to_dict,
    validate,
)

__all__ = [
    "CaseResult",
    "MetricDelta",
    "MetricSpec",
    "OracleSpec",
    "PipelineSpec",
    "ValidationCapabilities",
    "ValidationRequest",
    "ValidationResult",
    "ValidationStatus",
    "capabilities",
    "result_to_dict",
    "validate",
]
