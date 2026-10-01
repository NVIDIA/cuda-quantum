# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Callable CUDA-Q Logical resource-estimation namespace."""

from __future__ import annotations

import sys
import types as _py_types

from .logical import LogicalEstimate, LogicalProfile, logical_counts
from .static import count
from .analytical import analytical
from .schedule import scheduled
from .types import (
    EvidencePolicy,
    FabricCounts,
    FabricEstimate,
    FailureBudget,
    MissingEvidence,
    RetryDemand,
    Scaling,
    ScheduleTermination,
    ScheduleEstimate,
    Tier,
)


class _CallableEstimate(_py_types.ModuleType):

    def __call__(self, build, *, tier=Tier.STATIC, **kwargs):
        if isinstance(tier, str):
            try:
                tier = Tier[tier.upper()]
            except KeyError as exc:
                raise ValueError(f"unknown estimation tier {tier!r}") from exc
        if tier is Tier.LOGICAL:
            if kwargs:
                raise TypeError("Tier.LOGICAL does not accept physical options")
            return logical_counts(_as_build(build))
        if tier is Tier.STATIC:
            if kwargs:
                raise TypeError("Tier.STATIC does not accept physical options")
            return count(_as_build(build))
        if tier is Tier.ANALYTICAL:
            return analytical(_as_build(build), **kwargs)
        if tier is Tier.SCHEDULE:
            return scheduled(build, **kwargs)
        raise ValueError(f"unsupported estimation tier {tier!r}")


def _as_build(value):
    """Materialize authoring definitions at an estimator's natural profile.

    Estimation is a user-facing terminal operation, just like target
    preparation, so requiring an otherwise redundant ``cudaq.logical.compile``
    at every
    call site only leaks the compiler's staging API.  Already-built values are
    preserved; definitions are compiled through their normal default pipeline.
    The individual estimator still owns profile validation and therefore emits
    the useful P0/P2 diagnostic if the selected tier and definition disagree.
    """
    from ..compiler import Build, compile

    return value if isinstance(value, Build) else compile(value)


sys.modules[__name__].__class__ = _CallableEstimate

# The public ``schedule`` alias deliberately shadows the internal submodule
# attribute while keeping the P3 estimator easy to discover.
schedule = scheduled

__all__ = [
    "Tier",
    "LogicalProfile",
    "LogicalEstimate",
    "FabricCounts",
    "FabricEstimate",
    "FailureBudget",
    "Scaling",
    "EvidencePolicy",
    "MissingEvidence",
    "RetryDemand",
    "ScheduleTermination",
    "ScheduleEstimate",
    "logical_counts",
    "count",
    "analytical",
    "scheduled",
    "schedule",
]
