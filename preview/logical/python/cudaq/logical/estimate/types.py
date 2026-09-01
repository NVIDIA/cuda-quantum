# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, ClassVar, Mapping


class Tier(Enum):
    LOGICAL = auto()
    STATIC = auto()


def estimate_to_dict(value):
    """Project an in-scope estimate value to plain data."""

    if is_dataclass(value):
        return {
            item.name: estimate_to_dict(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): estimate_to_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [estimate_to_dict(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{type(value).__name__} is not an estimate POD value")


@dataclass(frozen=True, slots=True, kw_only=True)
class _EstimateResult:
    """Shared POD projection for public non-sampling estimate results."""

    annotation_tier: ClassVar[Tier]
    build_root: str = ""
    build_sha256: str = ""

    def to_dict(self):
        return estimate_to_dict(self)

    @classmethod
    def from_annotations(cls, annotations: Mapping[str, Any]):
        """Rehydrate this estimate from CUDA-Q ``EstimateResult`` metadata."""

        if not isinstance(annotations, Mapping):
            raise TypeError("annotations must be a mapping")
        tier_name = cls.annotation_tier.name
        try:
            value = annotations[tier_name]
        except KeyError as exc:
            raise KeyError(
                f"annotations do not contain the {tier_name!r} estimate"
            ) from exc
        if not isinstance(value, Mapping):
            raise TypeError(f"{tier_name!r} annotation must be a JSON object")
        return cls(**value)


def frozen_mapping(values):
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class FabricCounts(_EstimateResult):
    annotation_tier: ClassVar[Tier] = Tier.STATIC

    operation_counts: Mapping[str, int]
    gadget_calls: Mapping[str, int]
    protocol_calls: Mapping[str, int]
    resource_requests: Mapping[str, int]
    resource_stream_requests: Mapping[str, int]
    success_count: int
    syndrome_rounds: int
    patches_peak: int
    logical_qubits_peak: int
    hierarchy_depths: Mapping[str, int]
    source_stage: str
    source_facets: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
                "operation_counts",
                "gadget_calls",
                "protocol_calls",
                "resource_requests",
                "resource_stream_requests",
                "hierarchy_depths",
        ):
            object.__setattr__(self, field, frozen_mapping(getattr(self,
                                                                   field)))
        object.__setattr__(self, "source_facets", tuple(self.source_facets))

    @property
    def total_operations(self) -> int:
        structural = {
            "call",
            "establish_support",
            "establish_topological_record",
            "map_children",
            "relocate",
            "repeat",
        }
        return sum(count for name, count in self.operation_counts.items()
                   if name not in structural)
