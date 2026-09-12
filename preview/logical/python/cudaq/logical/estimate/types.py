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
import math
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from ..stages import Facet, Stage

_SCHEDULE_ESTIMATE_CREATION_TOKEN = object()


class Tier(Enum):
    LOGICAL = auto()
    STATIC = auto()
    ANALYTICAL = auto()
    SCHEDULE = auto()


class ScheduleTermination(str, Enum):
    """Whether Tier-3 costing follows runtime aborts or the full workload."""

    PROGRAM = "program"
    FULL_WORKLOAD = "full_workload"


def estimate_to_dict(value):
    """Project an in-scope estimate value to CUDA-Q annotation data."""

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


class _EstimateResult:
    """Shared CUDA-Q annotation bridge for public estimate results."""

    annotation_tier: ClassVar[Tier]

    def to_dict(self):
        return estimate_to_dict(self)

    @classmethod
    def from_annotations(cls, annotations: Mapping[str, Any]):
        """Rehydrate this result from CUDA-Q ``EstimateResult`` metadata."""

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
    success_count: int
    syndrome_rounds: int
    patches_peak: int
    logical_qubits_peak: int
    hierarchy_depths: Mapping[str, int]
    source_stage: str
    source_facets: tuple[str, ...]
    build_root: str = ""
    build_sha256: str = ""

    def __post_init__(self) -> None:
        for field in (
                "operation_counts",
                "gadget_calls",
                "protocol_calls",
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


@dataclass(frozen=True, slots=True)
class FailureBudget:
    total: float

    def __post_init__(self) -> None:
        if not 0.0 < self.total <= 1.0:
            raise ValueError("failure budget total must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class Scaling:
    prefactor: float = 0.1
    threshold: float = 0.01

    def __post_init__(self) -> None:
        if (not math.isfinite(self.prefactor) or self.prefactor < 0.0 or
                not math.isfinite(self.threshold) or self.threshold <= 0.0):
            raise ValueError(
                "scaling prefactor must be finite and nonnegative and "
                "threshold must be finite and positive")

    def p_logical(self, distance: int, p_phys: float) -> float:
        if distance <= 0:
            raise ValueError("distance must be positive")
        if not 0.0 <= p_phys <= 1.0:
            raise ValueError("physical error probability must lie in [0, 1]")
        return min(
            1.0,
            self.prefactor * (p_phys / self.threshold)**((distance + 1) / 2),
        )


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    require_established: bool = False

    @classmethod
    def require_established_distance(cls):
        return cls(require_established=True)


@dataclass(frozen=True, slots=True)
class RetryDemand:
    """Expected and bounded demand derived from one named retry attempt."""

    attempt: str
    occurrences: int
    success_probability: float
    max_attempts: int
    expected_attempts: float
    exhaustion_probability: float
    resource_requests_per_attempt: Mapping[str, int]
    expected_resource_requests: Mapping[str, float]
    maximum_resource_requests: Mapping[str, int]
    action_site: str | None = None
    effective_angle: float | None = None
    precision: float | None = None
    synthesis_sha256: str | None = None

    @property
    def completion_probability(self) -> float:
        if self.success_probability == 1.0:
            return 1.0
        return -math.expm1(
            self.max_attempts * math.log1p(-self.success_probability))

    def __post_init__(self) -> None:
        if not self.attempt:
            raise ValueError("retry demand attempt must be nonempty")
        if self.occurrences <= 0 or self.max_attempts <= 0:
            raise ValueError("retry demand counts must be positive")
        if not 0.0 < self.success_probability <= 1.0:
            raise ValueError("retry success probability must lie in (0, 1]")
        if not 1.0 <= self.expected_attempts <= self.max_attempts:
            raise ValueError(
                "retry expected attempts must lie in [1, max_attempts]")
        if not 0.0 <= self.exhaustion_probability <= 1.0:
            raise ValueError("retry exhaustion probability must lie in [0, 1]")
        if self.effective_angle is not None and not math.isfinite(
                self.effective_angle):
            raise ValueError("retry effective angle must be finite")
        if self.precision is not None and (not math.isfinite(self.precision) or
                                           self.precision <= 0.0):
            raise ValueError("retry precision must be finite and positive")
        if self.synthesis_sha256 is not None and (
                len(self.synthesis_sha256) != 64 or
                any(character not in "0123456789abcdef"
                    for character in self.synthesis_sha256)):
            raise ValueError("retry synthesis digest must be lowercase sha256")
        for field in (
                "resource_requests_per_attempt",
                "expected_resource_requests",
                "maximum_resource_requests",
        ):
            object.__setattr__(self, field, frozen_mapping(getattr(self,
                                                                   field)))


@dataclass(frozen=True, slots=True)
class FabricEstimate(_EstimateResult):
    annotation_tier: ClassVar[Tier] = Tier.ANALYTICAL

    counts: FabricCounts
    p_phys: float
    failure_budget: FailureBudget
    distance: int
    distance_status: str
    logical_error: float
    physical_qubits_peak: int
    wallclock: float
    cycle_time: float
    acceptance: float
    bottleneck: str
    budget_met: bool
    assumptions: tuple[str, ...]
    retry_demands: tuple[RetryDemand, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "retry_demands", tuple(self.retry_demands))

    @classmethod
    def from_annotations(cls, annotations: Mapping[str, Any]):
        """Rehydrate nested typed analytical evidence from CUDA-Q metadata."""

        if not isinstance(annotations, Mapping):
            raise TypeError("annotations must be a mapping")
        tier_name = cls.annotation_tier.name
        try:
            raw = annotations[tier_name]
        except KeyError as exc:
            raise KeyError(
                f"annotations do not contain the {tier_name!r} estimate"
            ) from exc
        if not isinstance(raw, Mapping):
            raise TypeError(f"{tier_name!r} annotation must be a JSON object")
        values = dict(raw)
        try:
            values["counts"] = FabricCounts(**values["counts"])
            values["failure_budget"] = FailureBudget(**values["failure_budget"])
            values["retry_demands"] = tuple(
                RetryDemand(**item) for item in values.get("retry_demands", ()))
        except (KeyError, TypeError) as exc:
            raise TypeError(
                f"{tier_name!r} annotation contains invalid nested evidence"
            ) from exc
        return cls(**values)


@dataclass(frozen=True, slots=True, init=False)
class ScheduleEstimate(_EstimateResult):
    """Immutable native-authenticated Tier-3 resource evidence."""

    annotation_tier = Tier.SCHEDULE

    event_count: int
    event_counts: Mapping[str, int]
    makespan_ns: float
    expected_makespan_ns: float
    maximum_makespan_ns: float
    active_resource_time_ns: float
    expected_active_resource_time_ns: float
    maximum_active_resource_time_ns: float
    active_physical_qubit_time_ns: float
    expected_active_physical_qubit_time_ns: float
    maximum_active_physical_qubit_time_ns: float
    physical_resources: int
    physical_qubits: int
    peak_concurrency: int
    peak_active_physical_qubits: int
    utilization: float
    expected_utilization: float
    maximum_utilization: float
    exhaustion_probability: float
    termination: ScheduleTermination
    bottleneck: str
    assumptions: tuple[str, ...]
    input_root: str
    schedule_symbol: str
    source_stage: Stage
    source_facets: tuple[Facet, ...]
    tier: Tier
    device_identity: str | None
    physical_model_identity: str
    operating_point_identity: str | None
    lower_tier_identity: str

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError("ScheduleEstimate cannot be constructed directly; use "
                        "cudaq.logical.estimate(..., tier=Tier.SCHEDULE)")

    @classmethod
    def from_annotations(cls, annotations: Mapping[str, Any]):
        raise TypeError(
            "ScheduleEstimate cannot be rehydrated from unauthenticated "
            "annotations; use cudaq.logical.estimate(..., tier=Tier.SCHEDULE)")

    @classmethod
    def _create(cls, *, _token=None, **values):
        if _token is not _SCHEDULE_ESTIMATE_CREATION_TOKEN:
            raise TypeError(
                "ScheduleEstimate creation is reserved for the native "
                "schedule estimator")
        field_names = tuple(field.name for field in fields(cls))
        missing = tuple(name for name in field_names if name not in values)
        extra = tuple(name for name in values if name not in field_names)
        if missing or extra:
            raise TypeError(
                "authenticated ScheduleEstimate fields are incomplete "
                f"(missing={missing!r}, extra={extra!r})")
        value = object.__new__(cls)
        for name in field_names:
            object.__setattr__(value, name, values[name])
        value._validate()
        return value

    def _validate(self) -> None:
        count_fields = (
            "event_count",
            "physical_resources",
            "physical_qubits",
            "peak_concurrency",
            "peak_active_physical_qubits",
        )
        for name in count_fields:
            value = getattr(self, name)
            if isinstance(value,
                          bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"schedule estimate {name} must be a nonnegative integer")

        if not isinstance(self.event_counts, Mapping):
            raise TypeError("schedule estimate event_counts must be a mapping")
        event_counts = {}
        for kind, count in self.event_counts.items():
            if not isinstance(kind, str) or not kind:
                raise ValueError(
                    "schedule estimate event_counts keys must be nonempty strings"
                )
            if (isinstance(count, bool) or not isinstance(count, int) or
                    count < 0):
                raise ValueError(
                    "schedule estimate event_counts values must be "
                    "nonnegative integers")
            event_counts[kind] = count
        if sum(event_counts.values()) != self.event_count:
            raise ValueError(
                "schedule estimate event_count must equal the sum of "
                "event_counts")

        metric_families = (
            (
                "makespan_ns",
                "expected_makespan_ns",
                "maximum_makespan_ns",
            ),
            (
                "active_resource_time_ns",
                "expected_active_resource_time_ns",
                "maximum_active_resource_time_ns",
            ),
            (
                "active_physical_qubit_time_ns",
                "expected_active_physical_qubit_time_ns",
                "maximum_active_physical_qubit_time_ns",
            ),
        )
        numeric_fields = tuple(
            name for family in metric_families for name in family) + (
                "utilization",
                "expected_utilization",
                "maximum_utilization",
                "exhaustion_probability",
            )
        for name in numeric_fields:
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value,
                                                          (int, float)) or
                    not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(
                    f"schedule estimate {name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        for first_name, expected_name, maximum_name in metric_families:
            first = getattr(self, first_name)
            expected = getattr(self, expected_name)
            maximum = getattr(self, maximum_name)
            if first > maximum or expected > maximum:
                raise ValueError(
                    "schedule estimate metric families must satisfy "
                    "first-attempt <= maximum and expected <= maximum")

        for name in (
                "utilization",
                "expected_utilization",
                "maximum_utilization",
                "exhaustion_probability",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"schedule estimate {name} must lie in [0, 1]")
        if not isinstance(self.termination, ScheduleTermination):
            raise TypeError(
                "schedule estimate termination must be a ScheduleTermination")
        if self.peak_concurrency > self.physical_resources:
            raise ValueError(
                "schedule estimate peak_concurrency cannot exceed provisioned "
                "physical_resources")
        if self.peak_active_physical_qubits > self.physical_qubits:
            raise ValueError(
                "schedule estimate peak_active_physical_qubits cannot exceed "
                "provisioned physical_qubits")

        utilization_metrics = (
            ("utilization", "active_resource_time_ns", "makespan_ns"),
            (
                "expected_utilization",
                "expected_active_resource_time_ns",
                "expected_makespan_ns",
            ),
            (
                "maximum_utilization",
                "maximum_active_resource_time_ns",
                "maximum_makespan_ns",
            ),
        )
        tolerance = 128.0 * math.ulp(1.0)
        for utilization_name, active_name, makespan_name in utilization_metrics:
            capacity = getattr(self, makespan_name) * self.physical_resources
            expected = (0.0 if capacity == 0.0 else getattr(self, active_name) /
                        capacity)
            if not math.isclose(
                    getattr(self, utilization_name),
                    expected,
                    rel_tol=tolerance,
                    abs_tol=tolerance,
            ):
                raise ValueError(
                    f"schedule estimate {utilization_name} must equal active "
                    "resource-time divided by scheduled capacity")

        if not isinstance(self.bottleneck, str) or not self.bottleneck:
            raise ValueError(
                "schedule estimate bottleneck must be a nonempty string")
        assumptions = tuple(self.assumptions)
        if (not assumptions or any(not isinstance(value, str) or not value
                                   for value in assumptions) or
                len(set(assumptions)) != len(assumptions)):
            raise ValueError(
                "schedule estimate assumptions must be unique nonempty strings")
        if (not isinstance(self.input_root, str) or not self.input_root or
                not isinstance(self.schedule_symbol, str) or
                not self.schedule_symbol):
            raise ValueError(
                "schedule estimate input root and schedule symbol must be nonempty"
            )
        if self.source_stage is not Stage.P3:
            raise ValueError("schedule estimate source_stage must be Stage.P3")
        source_facets = tuple(self.source_facets)
        if any(not isinstance(facet, Facet) for facet in source_facets):
            raise TypeError(
                "schedule estimate source_facets must contain Facet values")
        if (Facet.PHYSICAL_SCHEDULE not in source_facets or
                len(set(source_facets)) != len(source_facets)):
            raise ValueError(
                "schedule estimate source_facets must uniquely retain the "
                "physical-schedule facet")
        if self.tier is not Tier.SCHEDULE:
            raise ValueError("ScheduleEstimate tier must be Tier.SCHEDULE")
        if (not isinstance(self.physical_model_identity, str) or
                not self.physical_model_identity):
            raise ValueError(
                "schedule estimate physical_model_identity must be nonempty")
        if (not isinstance(self.lower_tier_identity, str) or
                not self.lower_tier_identity):
            raise ValueError(
                "schedule estimate lower_tier_identity must be nonempty")
        for name in (
                "device_identity",
                "operating_point_identity",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(
                    f"schedule estimate {name} must be a nonempty string or None"
                )
        object.__setattr__(self, "event_counts", frozen_mapping(event_counts))
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "source_facets", source_facets)


class MissingEvidence(ValueError):
    pass
