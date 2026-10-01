# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Typed compact P3 models for synchronous protocols and transport."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import Enum
from math import isfinite

from cudaq.logical.analysis.evidence import Provenance
from cudaq.logical.architecture.physical_definition import ResourceClass
from cudaq.logical.protocols.definition import ProtocolDefinition


def _positive_number(value, *, what: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not isfinite(float(value)) or float(value) <= 0.0):
        raise TypeError(f"{what} must be a finite positive number")
    return float(value)


def _positive_int(value, *, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{what} must be a positive int")
    return value


def _sha256(value: str, *, what: str, prefixed: bool = False) -> str:
    prefix = "sha256:" if prefixed else ""
    length = 71 if prefixed else 64
    if not isinstance(value, str):
        spelling = "canonical lowercase sha256 commitment" if prefixed else (
            "lowercase SHA-256 digest")
        raise ValueError(f"{what} must be a {spelling}")
    payload = value.removeprefix(prefix)
    if (len(value) != length or (prefixed and not value.startswith(prefix)) or
            any(character not in "0123456789abcdef" for character in payload)):
        spelling = "canonical lowercase sha256 commitment" if prefixed else (
            "lowercase SHA-256 digest")
        raise ValueError(f"{what} must be a {spelling}")
    return value


class InitiationIntervalSemantics(str, Enum):
    """Meaning of a compact model's latency/interval relationship."""

    PIPELINED = "pipelined"
    BACKPRESSURED = "backpressured"


@dataclass(frozen=True, slots=True)
class PhysicalResourceClaim:
    """One exact P3 resource-class slice.

    ``units`` is the number of members acquired from the slice by one
    occurrence.  Synchronous phases require the whole slice.  Transport may
    acquire fewer members, allowing the generic scheduler to choose a legal
    lane deterministically without adding another topology.
    """

    resource_class: ResourceClass
    offset: int = 0
    count: int | None = None
    units: int | None = None

    def __post_init__(self) -> None:
        resource = self.resource_class
        if not isinstance(resource, ResourceClass):
            raise TypeError(
                "PhysicalResourceClaim.resource_class must be a ResourceClass")
        if resource.name is None:
            raise ValueError(
                "PhysicalResourceClaim requires a named ResourceClass")
        if (isinstance(self.offset, bool) or not isinstance(self.offset, int) or
                self.offset < 0):
            raise TypeError(
                "PhysicalResourceClaim.offset must be a nonnegative int")
        count = resource.count - self.offset if self.count is None else self.count
        count = _positive_int(count, what="PhysicalResourceClaim.count")
        if self.offset + count > resource.count:
            raise ValueError(
                "PhysicalResourceClaim slice exceeds its resource class")
        units = count if self.units is None else self.units
        units = _positive_int(units, what="PhysicalResourceClaim.units")
        if units > count:
            raise ValueError(
                "PhysicalResourceClaim.units cannot exceed its slice count")
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "units", units)


def _reject_overlapping_claims(values, *, what: str) -> None:
    """Reject two declarations of the same physical member.

    Claims are authored as slices, but the scheduler reserves individual
    members.  Accepting intersecting slices would therefore let two distinct
    committed declarations collapse to one reservation after expansion.
    """

    by_resource = {}
    for claim in values:
        intervals = by_resource.setdefault(id(claim.resource_class), [])
        begin = claim.offset
        end = begin + claim.count
        if any(begin < other_end and other_begin < end
               for other_begin, other_end in intervals):
            raise ValueError(f"{what} has overlapping resource claims")
        intervals.append((begin, end))


@dataclass(frozen=True, slots=True)
class SpacetimePhase:
    """One ordered phase of a compact synchronous P3 plan."""

    name: str
    steps: int
    step_duration_cycles: float
    resources: tuple[PhysicalResourceClaim, ...]
    factories: tuple[object, ...] = ()
    after: tuple["SpacetimePhase | str", ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError(
                "SpacetimePhase.name must be a nonempty Python identifier")
        object.__setattr__(
            self, "steps", _positive_int(self.steps,
                                         what="SpacetimePhase.steps"))
        object.__setattr__(
            self,
            "step_duration_cycles",
            _positive_number(
                self.step_duration_cycles,
                what="SpacetimePhase.step_duration_cycles",
            ),
        )
        resources = tuple(self.resources)
        if not resources or any(not isinstance(value, PhysicalResourceClaim)
                                for value in resources):
            raise TypeError(
                "SpacetimePhase.resources must contain typed resource claims")
        if any(value.units != value.count for value in resources):
            raise ValueError(
                "synchronous phases must acquire each complete resource slice")
        _reject_overlapping_claims(resources, what="SpacetimePhase")
        factories = tuple(self.factories)
        from .definition import FactoryModel
        if any(not isinstance(value, FactoryModel) for value in factories):
            raise TypeError(
                "SpacetimePhase.factories must contain FactoryModel values")
        if len({id(value) for value in factories}) != len(factories):
            raise ValueError("SpacetimePhase factory claims must be unique")
        after = tuple(value.name if isinstance(value, SpacetimePhase) else value
                      for value in self.after)
        if any(not isinstance(value, str) or not value.isidentifier()
               for value in after):
            raise TypeError(
                "SpacetimePhase.after must contain phases or phase names")
        if len(set(after)) != len(after) or self.name in after:
            raise ValueError(
                "SpacetimePhase dependencies must be unique and non-self")
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "factories", factories)
        object.__setattr__(self, "after", after)


_SPACETIME_CHARACTERIZATION_AUTH = object()
_TRANSPORT_CHARACTERIZATION_AUTH = object()


@dataclass(frozen=True, slots=True)
class SpacetimePlanCharacterization:
    """Opaque compiler evidence for one synchronous compact P3 plan."""

    source_protocol: str
    source_protocol_sha256: str
    selected_protocol_sha256: str
    objective_sha256: str
    boundary_sha256: str
    architecture: str
    architecture_sha256: str
    operating_point: str
    timing_source: str | None
    timing_profile: tuple[tuple[str, float], ...]
    code_distances: tuple[int, ...]
    latency_cycles: float
    initiation_interval_cycles: float
    provider: str
    provider_version: str
    derivation: str
    derivation_version: int
    model_sha256: str
    build_sha256: str
    schedule_sha256: str
    _auth: InitVar[object] = None

    def __post_init__(self, _auth) -> None:
        if _auth is not _SPACETIME_CHARACTERIZATION_AUTH:
            raise TypeError(
                "SpacetimePlanCharacterization is an opaque compiler artifact; "
                "obtain it from cudaq.logical.compiler.spacetime_plan_model()")
        for name in ("source_protocol", "architecture", "operating_point",
                     "provider", "provider_version", "derivation"):
            if not isinstance(getattr(self, name), str) or not getattr(
                    self, name):
                raise ValueError(
                    f"SpacetimePlanCharacterization.{name} must be nonempty")
        for name in ("source_protocol_sha256", "selected_protocol_sha256",
                     "objective_sha256", "boundary_sha256",
                     "architecture_sha256"):
            _sha256(getattr(self, name),
                    what=(f"SpacetimePlanCharacterization.{name}"),
                    prefixed=True)
        for name in ("model_sha256", "build_sha256", "schedule_sha256"):
            _sha256(getattr(self, name),
                    what=(f"SpacetimePlanCharacterization.{name}"))
        for name in ("latency_cycles", "initiation_interval_cycles"):
            object.__setattr__(
                self, name,
                _positive_number(getattr(self, name),
                                 what=f"SpacetimePlanCharacterization.{name}"))
        if (isinstance(self.derivation_version, bool) or
                not isinstance(self.derivation_version, int) or
                self.derivation_version <= 0):
            raise TypeError(
                "SpacetimePlanCharacterization.derivation_version must be positive"
            )
        distances = tuple(self.code_distances)
        if distances != tuple(sorted(set(distances))) or any(
                isinstance(value, bool) or not isinstance(value, int) or
                value <= 0 for value in distances):
            raise ValueError(
                "SpacetimePlanCharacterization.code_distances must be sorted "
                "unique positive integers")
        object.__setattr__(self, "code_distances", distances)
        object.__setattr__(
            self, "timing_profile",
            _timing_profile(
                self.timing_profile,
                what="SpacetimePlanCharacterization.timing_profile"))
        if self.timing_source is not None and (not isinstance(
                self.timing_source, str) or not self.timing_source):
            raise ValueError(
                "SpacetimePlanCharacterization.timing_source must be nonempty or None"
            )


def _timing_profile(values, *, what: str):
    result = tuple((str(name), float(value)) for name, value in values)
    if result != tuple(sorted(result)) or len({name for name, _ in result
                                              }) != len(result):
        raise ValueError(f"{what} must be sorted with unique keys")
    if any(not name or not isfinite(value) or value < 0.0
           for name, value in result):
        raise ValueError(f"{what} must contain finite nonnegative named values")
    return result


def _compiled_spacetime_evidence(value) -> Provenance:
    return Provenance(
        "computation",
        f"qlx.spacetime-plan-characterization/v1:{value.build_sha256}:"
        f"{value.schedule_sha256}:{value.model_sha256}",
    )


def _spacetime_characterization_from_verified_schedule(**values):
    return SpacetimePlanCharacterization(**values,
                                         _auth=_SPACETIME_CHARACTERIZATION_AUTH)


@dataclass(frozen=True, slots=True)
class SpacetimePlanModel:
    """Compact synchronous physical realization of one exact P2 protocol."""

    protocol: ProtocolDefinition
    latency_cycles: float
    initiation_interval_cycles: float
    phases: tuple[SpacetimePhase, ...]
    evidence: Provenance
    code_distances: tuple[int, ...] = ()
    interval_semantics: InitiationIntervalSemantics = (
        InitiationIntervalSemantics.PIPELINED)
    policy: str = "guaranteed"
    characterization: SpacetimePlanCharacterization | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, ProtocolDefinition):
            raise TypeError(
                "SpacetimePlanModel.protocol must be a ProtocolDefinition")
        for name in ("latency_cycles", "initiation_interval_cycles"):
            object.__setattr__(
                self, name,
                _positive_number(getattr(self, name),
                                 what=f"SpacetimePlanModel.{name}"))
        try:
            semantics = InitiationIntervalSemantics(self.interval_semantics)
        except ValueError as error:
            raise ValueError(
                "SpacetimePlanModel.interval_semantics must be typed"
            ) from error
        if (self.initiation_interval_cycles > self.latency_cycles and
                semantics is not InitiationIntervalSemantics.BACKPRESSURED):
            raise ValueError(
                "initiation interval greater than one-shot latency requires "
                "backpressured interval semantics")
        object.__setattr__(self, "interval_semantics", semantics)
        phases = tuple(self.phases)
        if not phases or any(
                not isinstance(value, SpacetimePhase) for value in phases):
            raise TypeError("SpacetimePlanModel.phases must contain phases")
        names = tuple(value.name for value in phases)
        if len(set(names)) != len(names):
            raise ValueError("SpacetimePlanModel phase names must be unique")
        seen = set()
        for phase in phases:
            if any(dependency not in seen for dependency in phase.after):
                raise ValueError(
                    "SpacetimePlanModel dependencies must name earlier phases")
            seen.add(phase.name)
        distances = tuple(self.code_distances)
        if distances != tuple(sorted(set(distances))) or any(
                isinstance(value, bool) or not isinstance(value, int) or
                value <= 0 for value in distances):
            raise ValueError(
                "SpacetimePlanModel.code_distances must be sorted unique "
                "positive integers")
        if not isinstance(self.evidence, Provenance):
            raise TypeError("SpacetimePlanModel.evidence must be Provenance")
        if self.policy not in {"guaranteed", "single_shot"}:
            raise ValueError(
                "SpacetimePlanModel policy must be guaranteed or single_shot")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "code_distances", distances)
        characterization = self.characterization
        if characterization is not None:
            if not isinstance(characterization, SpacetimePlanCharacterization):
                raise TypeError(
                    "SpacetimePlanModel.characterization must be opaque compiler evidence"
                )
            from cudaq.logical.compiler.component_identity import protocol_contract
            if characterization.source_protocol_sha256 != protocol_contract(
                    self.protocol)["source_sha256"]:
                raise ValueError(
                    "characterized spacetime model has a detached source protocol"
                )
            if (self.latency_cycles != characterization.latency_cycles or
                    self.initiation_interval_cycles
                    != characterization.initiation_interval_cycles):
                raise ValueError(
                    "characterized spacetime model must retain exact compiler timing"
                )
            if distances != characterization.code_distances:
                raise ValueError(
                    "characterized spacetime model must retain exact code distances"
                )
            if self.evidence != _compiled_spacetime_evidence(characterization):
                raise ValueError(
                    "characterized spacetime model must retain exact compiler evidence"
                )
            from cudaq.logical.compiler.component_identity import spacetime_model_sha256
            if spacetime_model_sha256(self) != characterization.model_sha256:
                raise ValueError(
                    "characterized spacetime model must retain exact phases and claims"
                )


@dataclass(frozen=True, slots=True)
class TransportCharacterization:
    """Opaque compiler evidence for one selected P1/P2/P3 channel path."""

    channel: str
    channel_sha256: str
    realization_sha256: str
    protocol_sha256: str
    selected_protocol_sha256: str
    binding_sha256: str
    architecture: str
    architecture_sha256: str
    operating_point: str
    timing_source: str | None
    timing_profile: tuple[tuple[str, float], ...]
    latency_cycles: float
    initiation_interval_cycles: float
    model_sha256: str
    provider: str
    provider_version: str
    build_sha256: str
    schedule_sha256: str
    transfer_events: tuple[str, ...]
    _auth: InitVar[object] = None

    def __post_init__(self, _auth) -> None:
        if _auth is not _TRANSPORT_CHARACTERIZATION_AUTH:
            raise TypeError(
                "TransportCharacterization is an opaque compiler artifact; "
                "obtain it from cudaq.logical.compiler.transport_model()")
        for name in ("channel", "architecture", "operating_point", "provider",
                     "provider_version"):
            if not isinstance(getattr(self, name), str) or not getattr(
                    self, name):
                raise ValueError(
                    f"TransportCharacterization.{name} must be nonempty")
        for name in ("channel_sha256", "realization_sha256", "protocol_sha256",
                     "selected_protocol_sha256", "binding_sha256",
                     "architecture_sha256"):
            _sha256(getattr(self, name),
                    what=f"TransportCharacterization.{name}",
                    prefixed=True)
        for name in ("model_sha256", "build_sha256", "schedule_sha256"):
            _sha256(getattr(self, name),
                    what=f"TransportCharacterization.{name}")
        for name in ("latency_cycles", "initiation_interval_cycles"):
            object.__setattr__(
                self, name,
                _positive_number(getattr(self, name),
                                 what=f"TransportCharacterization.{name}"))
        events = tuple(self.transfer_events)
        if not events or any(not isinstance(value, str) or not value
                             for value in events) or len(
                                 set(events)) != len(events):
            raise ValueError(
                "TransportCharacterization.transfer_events must be nonempty and unique"
            )
        object.__setattr__(self, "transfer_events", events)
        object.__setattr__(
            self, "timing_profile",
            _timing_profile(self.timing_profile,
                            what="TransportCharacterization.timing_profile"))
        if self.timing_source is not None and (not isinstance(
                self.timing_source, str) or not self.timing_source):
            raise ValueError(
                "TransportCharacterization.timing_source must be nonempty or None"
            )


def _compiled_transport_evidence(value) -> Provenance:
    return Provenance(
        "computation",
        f"qlx.transport-characterization/v1:{value.build_sha256}:"
        f"{value.schedule_sha256}:{value.model_sha256}",
    )


def _transport_characterization_from_verified_schedule(**values):
    return TransportCharacterization(**values,
                                     _auth=_TRANSPORT_CHARACTERIZATION_AUTH)


@dataclass(frozen=True, slots=True)
class TransportModel:
    """Physical performance model for one bound typed channel realization."""

    latency_cycles: float
    initiation_interval_cycles: float
    resources: tuple[PhysicalResourceClaim, ...]
    endpoint_occupancy: tuple[int, int]
    evidence: Provenance
    interval_semantics: InitiationIntervalSemantics = (
        InitiationIntervalSemantics.PIPELINED)
    policy: str = "guaranteed"
    characterization: TransportCharacterization | None = None

    def __post_init__(self) -> None:
        for name in ("latency_cycles", "initiation_interval_cycles"):
            object.__setattr__(
                self, name,
                _positive_number(getattr(self, name),
                                 what=f"TransportModel.{name}"))
        try:
            semantics = InitiationIntervalSemantics(self.interval_semantics)
        except ValueError as error:
            raise ValueError(
                "TransportModel.interval_semantics must be typed") from error
        if (self.initiation_interval_cycles > self.latency_cycles and
                semantics is not InitiationIntervalSemantics.BACKPRESSURED):
            raise ValueError(
                "initiation interval greater than one-shot latency requires "
                "backpressured interval semantics")
        object.__setattr__(self, "interval_semantics", semantics)
        resources = tuple(self.resources)
        if not resources or any(not isinstance(value, PhysicalResourceClaim)
                                for value in resources):
            raise TypeError(
                "TransportModel.resources must contain typed claims")
        _reject_overlapping_claims(resources, what="TransportModel")
        occupancy = tuple(self.endpoint_occupancy)
        if len(occupancy) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) or
                value <= 0 for value in occupancy):
            raise TypeError(
                "TransportModel.endpoint_occupancy requires two positive ints")
        if not isinstance(self.evidence, Provenance):
            raise TypeError("TransportModel.evidence must be Provenance")
        if self.policy not in {"guaranteed", "single_shot"}:
            raise ValueError(
                "TransportModel policy must be guaranteed or single_shot")
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "endpoint_occupancy", occupancy)
        characterization = self.characterization
        if characterization is not None:
            if not isinstance(characterization, TransportCharacterization):
                raise TypeError(
                    "TransportModel.characterization must be opaque compiler evidence"
                )
            if (self.latency_cycles != characterization.latency_cycles or
                    self.initiation_interval_cycles
                    != characterization.initiation_interval_cycles):
                raise ValueError(
                    "characterized transport model must retain exact compiler timing"
                )
            if self.evidence != _compiled_transport_evidence(characterization):
                raise ValueError(
                    "characterized transport model must retain exact compiler evidence"
                )
            from cudaq.logical.compiler.component_identity import transport_model_sha256
            if transport_model_sha256(self) != characterization.model_sha256:
                raise ValueError(
                    "characterized transport model must retain exact resource claims"
                )


__all__ = [
    "InitiationIntervalSemantics",
    "PhysicalResourceClaim",
    "SpacetimePhase",
    "SpacetimePlanCharacterization",
    "SpacetimePlanModel",
    "TransportCharacterization",
    "TransportModel",
]
