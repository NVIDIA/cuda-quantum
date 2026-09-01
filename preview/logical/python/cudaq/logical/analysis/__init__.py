# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Resource-estimation and evidence APIs."""

from importlib import import_module

from .evidence import Provenance, citation, computation, report, user_assertion
from ..estimate import FabricCounts, LogicalProfile, Tier, count, logical_counts


def estimate(build, *, tier=Tier.STATIC, **kwargs):
    """Run the canonical tiered resource estimator.

    The callable implementation remains in the dependency-neutral
    :mod:`cudaq.logical.estimate` package.  This analysis-owned entry point keeps normal
    authoring code on the canonical namespace without duplicating estimator
    dispatch or changing its validation behavior.
    """

    estimator = import_module("..estimate", __package__)
    return estimator(build, tier=tier, **kwargs)


__all__ = [
    "FabricCounts",
    "LogicalProfile",
    "Tier",
    "estimate",
    "logical_counts",
    "count",
    "Provenance",
    "citation",
    "user_assertion",
    "report",
    "computation",
]
