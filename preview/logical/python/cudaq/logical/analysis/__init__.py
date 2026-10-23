# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Resource estimation, gadget EDEM analysis, and evidence APIs."""

from importlib import import_module

from .evidence import Provenance, citation, computation, report, user_assertion
from .edem import (ElementaryFault, FaultSet, PauliInsertion, derive_edem,
                   standard_depolarizing_faults)
from ..estimate import (
    EvidencePolicy,
    FabricCounts,
    FabricEstimate,
    FailureBudget,
    LogicalProfile,
    MissingEvidence,
    RetryDemand,
    Scaling,
    ScheduleEstimate,
    Tier,
    analytical,
    count,
    logical_counts,
    schedule,
    scheduled,
)


def estimate(build, *, tier=Tier.STATIC, **kwargs):
    """Run the canonical tiered resource estimator.

    The callable implementation remains in the dependency-neutral
    :mod:`cudaq.logical.estimate` package. This analysis-owned entry point keeps normal
    authoring code on the canonical namespace without duplicating estimator
    dispatch or changing its validation behavior.
    """

    estimator = import_module("cudaq.logical.estimate")
    return estimator(build, tier=tier, **kwargs)


__all__ = [
    "ElementaryFault",
    "FaultSet",
    "PauliInsertion",
    "derive_edem",
    "standard_depolarizing_faults",
    "EvidencePolicy",
    "FabricCounts",
    "FabricEstimate",
    "FailureBudget",
    "LogicalProfile",
    "MissingEvidence",
    "RetryDemand",
    "Scaling",
    "ScheduleEstimate",
    "Tier",
    "estimate",
    "logical_counts",
    "count",
    "analytical",
    "scheduled",
    "schedule",
    "Provenance",
    "citation",
    "user_assertion",
    "report",
    "computation",
]
