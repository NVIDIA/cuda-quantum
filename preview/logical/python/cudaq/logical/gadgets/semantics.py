# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from inspect import Signature, signature
from math import isfinite
from types import MappingProxyType, NoneType
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Mapping,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from cudaq.logical.programs.binding import (
    LogicalPortRef,
    ObjectiveOperandRef,
)
from cudaq.logical._core.immutable import ImmutableValue

EncodingT = TypeVar("EncodingT")

from .interface import BlockEndpoint, OutcomeRole, patch
from .records import (
    InputSyndromeRef,
    ProfileParity,
    ProfileVectorExpr,
    RecordParity,
    RecordRef,
    RecordVectorParity,
)


def _parity(value):
    if isinstance(value, ProfileParity):
        return value
    if isinstance(value, RecordRef):
        return RecordParity((value,))
    if isinstance(value, RecordParity):
        return value
    if isinstance(value, InputSyndromeRef):
        return ProfileParity(input_syndromes=(value,))
    raise TypeError(
        "expected a gadget record, incoming syndrome, or profile parity")


@dataclass(frozen=True, slots=True)
class SuccessPredicate:
    parity: RecordParity | RecordRef | ProfileParity

    def __post_init__(self) -> None:
        object.__setattr__(self, "parity",
                           ProfileParity.from_value(_parity(self.parity)))


@dataclass(frozen=True, slots=True)
class OutputSyndromeAssignment:
    """Compiler-normalized scalar row for one output block endpoint.

    Authors bind whole ``endpoint.syndrome`` bundles.  This scalar form exists
    only after profile-graph normalization and is intentionally not exported
    from the top-level user surface.
    """

    endpoint: BlockEndpoint
    index: int
    parity: RecordParity | RecordRef | InputSyndromeRef | ProfileParity | bool

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint,
                          BlockEndpoint) or self.endpoint.side != "output":
            raise TypeError(
                "output syndrome target must be an output BlockEndpoint")
        if not isinstance(self.index, int) or isinstance(
                self.index, bool) or self.index < 0:
            raise TypeError("output syndrome index must be a nonnegative int")
        object.__setattr__(self, "parity",
                           ProfileParity.from_value(self.parity))

    @property
    def port(self) -> BlockEndpoint:
        return self.endpoint


class RetryExhaustion(str, Enum):
    """Behavior when a bounded retry exhausts its attempt budget."""

    REPORT_FAILURE = "report_failure"
    ABORT = "abort"
    RETURN_LAST = "return_last"

    @classmethod
    def from_value(cls, value) -> "RetryExhaustion":
        if isinstance(value, cls):
            return value
        raise TypeError("retry exhaustion must be a RetryExhaustion value")


class CommitPointKind(str, Enum):
    BEFORE_OUTPUT = "before_output"
    PACK_RESOURCE = "pack_resource"


@dataclass(frozen=True, slots=True)
class CommitPoint:
    """Typed boundary before which an attempt may be replayed safely."""

    kind: CommitPointKind
    output: BlockEndpoint | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CommitPointKind):
            raise TypeError("CommitPoint.kind must be a CommitPointKind")
        if self.kind is CommitPointKind.BEFORE_OUTPUT:
            if self.output is not None and not isinstance(
                    self.output, BlockEndpoint):
                raise TypeError(
                    "before-output commit points require a BlockEndpoint")
        elif self.output is not None:
            raise ValueError(
                "resource-output commit points do not name a block endpoint")

    def to_ir(self) -> str:
        if self.kind is CommitPointKind.BEFORE_OUTPUT and self.output is not None:
            return f"before_output:{self.output.name}"
        return self.kind.value


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Normalized immutable policy for one bounded protocol retry."""

    max_attempts: int
    exhaustion: RetryExhaustion = RetryExhaustion.REPORT_FAILURE
    commit_point: CommitPoint | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.max_attempts, int) or
                isinstance(self.max_attempts, bool) or self.max_attempts <= 0):
            raise TypeError("retry max_attempts must be a positive Python int")
        object.__setattr__(self, "exhaustion",
                           RetryExhaustion.from_value(self.exhaustion))
        if self.commit_point is not None and not isinstance(
                self.commit_point, CommitPoint):
            raise TypeError(
                "retry commit_point must be a cudaq.logical.CommitPoint")


def before_output(output: BlockEndpoint | None = None) -> CommitPoint:
    return CommitPoint(CommitPointKind.BEFORE_OUTPUT, output)


def before_resource_output() -> CommitPoint:
    return CommitPoint(CommitPointKind.PACK_RESOURCE)


@dataclass(frozen=True, slots=True)
class OutcomeSyndromeTerm:
    """One typed incoming-syndrome term in an objective outcome row.

    ``port`` is the input endpoint ordinal in the gadget interface, not its
    diagnostic Python name. ``index`` is the component in that endpoint's
    effective syndrome bundle.
    """

    port: int
    index: int

    def __post_init__(self) -> None:
        for label, value in (("port", self.port), ("index", self.index)):
            if not isinstance(value, int) or isinstance(value,
                                                        bool) or value < 0:
                raise TypeError(
                    f"OutcomeSyndromeTerm.{label} must be a nonnegative int")


@dataclass(frozen=True, slots=True)
class OutcomeMap:
    """Total GF(2) affine map from realization records to objective outcomes.

    Columns are the named realization records in ``records`` order; row ``i``
    defines objective outcome bit ``i`` as the record parity, that row's typed
    incoming-syndrome terms, and ``constants[i]``. ``roles[i]`` identifies
    whether the objective result is an ordinary application result or a
    zero-on-accept success mismatch.
    """

    records: tuple[str, ...]
    matrix: Any
    constants: tuple[int, ...] = ()
    input_syndromes: tuple[tuple[OutcomeSyndromeTerm, ...], ...] = ()
    roles: tuple[tuple[OutcomeRole, ...], ...] = ()

    def __post_init__(self) -> None:
        from cudaq.logical.algebra.gf2 import (
            GF2Matrix,
            _normalize_binary_values,
        )

        records = tuple(self.records)
        if any(not isinstance(name, str) or not name for name in records):
            raise ValueError("OutcomeMap records must be nonempty record names")
        if len(set(records)) != len(records):
            raise ValueError("OutcomeMap record names must be unique")
        if not isinstance(self.matrix, GF2Matrix):
            raise TypeError(
                "OutcomeMap.matrix must be a cudaq.logical.GF2Matrix")
        if self.matrix.ncols != len(records):
            raise ValueError(
                f"OutcomeMap matrix width {self.matrix.ncols} does not match "
                f"its {len(records)} named record(s)")
        constants = _normalize_binary_values(self.constants,
                                             what="OutcomeMap constants")
        if not constants:
            constants = (0,) * self.matrix.nrows
        if len(constants) != self.matrix.nrows:
            raise ValueError(
                "OutcomeMap constants must contain one bit per outcome row")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "constants", constants)
        input_syndromes = tuple(
            tuple(value if isinstance(value, OutcomeSyndromeTerm
                                     ) else OutcomeSyndromeTerm(*value)
                  for value in row)
            for row in self.input_syndromes)
        if not input_syndromes:
            input_syndromes = ((),) * self.matrix.nrows
        if len(input_syndromes) != self.matrix.nrows:
            raise ValueError(
                "OutcomeMap input_syndromes must contain one term sequence "
                "per outcome row")
        for row in input_syndromes:
            if len(set(row)) != len(row):
                raise ValueError(
                    "OutcomeMap input-syndrome terms must be duplicate-free")
        object.__setattr__(self, "input_syndromes", input_syndromes)

        roles = tuple(
            (value,) if isinstance(value, OutcomeRole) else tuple(value)
            for value in self.roles)
        if not roles:
            roles = ((OutcomeRole.RESULT,),) * self.matrix.nrows
        if len(roles) != self.matrix.nrows:
            raise ValueError(
                "OutcomeMap roles must contain one role sequence per outcome row"
            )
        supported_roles = {
            OutcomeRole.RESULT,
            OutcomeRole.SUCCESS,
        }
        normalized_roles = []
        for row in roles:
            if not row:
                raise ValueError("OutcomeMap row roles must be nonempty")
            if any(not isinstance(role, OutcomeRole) for role in row):
                raise TypeError(
                    "OutcomeMap row roles must be OutcomeRole values")
            if len(set(row)) != len(row) or any(
                    role not in supported_roles for role in row):
                raise ValueError(
                    "OutcomeMap row roles must be duplicate-free and contain "
                    "only 'result' or 'success'")
            normalized_roles.append(row)
        object.__setattr__(self, "roles", tuple(normalized_roles))

    @property
    def outcome_count(self) -> int:
        return self.matrix.nrows

    def indices_for(self, role: OutcomeRole) -> tuple[int, ...]:
        if not isinstance(role, OutcomeRole):
            raise TypeError("OutcomeMap role must be an OutcomeRole value")
        if role not in {
                OutcomeRole.RESULT,
                OutcomeRole.SUCCESS,
        }:
            raise ValueError("OutcomeMap role must be result or success")
        return tuple(
            index for index, roles in enumerate(self.roles) if role in roles)


class ProfileSemanticError(ValueError):
    """A semantic claim in a GadgetProfile is false or incomplete."""


def _scalar_profile_parities(expressions):
    """Expand profile expressions into their canonical ordered scalar rows."""

    for expression in expressions:
        parity = expression.parity
        if isinstance(parity, RecordVectorParity):
            yield from (
                ProfileParity(records=row.records) for row in parity.rows())
        elif isinstance(parity, ProfileVectorExpr):
            yield from parity.rows
        else:
            yield ProfileParity.from_value(parity)


def _outcome_role_parities(gadget: "GadgetDefinition", role: OutcomeRole):
    """Derive the exact ordered profile rows owned by one OutcomeMap role."""

    outcome_map = gadget.outcome_map
    if outcome_map is None:
        return ()
    indices = outcome_map.indices_for(role)
    return tuple(
        ProfileParity(
            records=tuple(
                RecordRef(gadget, record_name)
                for record_name, bit in zip(outcome_map.records, row)
                if bit),
            input_syndromes=tuple(
                InputSyndromeRef(gadget.interface.inputs[term.port], term.index)
                for term in outcome_map.input_syndromes[index]),
            constant=bool(constant),
        )
        for index, (row, constant) in enumerate(
            zip(outcome_map.matrix.rows, outcome_map.constants))
        if index in indices)


def _reconcile_profile_role(gadget, profile_name, role, declared):
    """Validate or derive one OutcomeMap-owned profile family.

    Once a profile declares any row in that family, the declaration must
    reproduce the complete ordered affine table. Omitted rows are derived
    immediately so every constructed profile has one canonical representation.
    """

    declared = tuple(declared)
    derived = _outcome_role_parities(gadget, role)
    if not derived:
        return declared
    if declared and len(declared) != len(derived):
        raise ProfileSemanticError(
            f"profile {profile_name!r} declares {len(declared)} {role.value} "
            f"row(s), but the gadget outcome map defines {len(derived)} "
            f"authoritative {role.value} row(s)")
    for index, (profile_row, outcome_row) in enumerate(zip(declared, derived)):
        if profile_row != outcome_row:
            raise ProfileSemanticError(
                f"profile {role.value} row {index} disagrees with the gadget "
                f"outcome-map row {index}")
    return declared or derived


@dataclass(frozen=True, slots=True)
class _PredicateProvenance:
    """Typed affine meaning of one Boolean inside a protocol trace.

    ``parity`` is the exact GadgetSpec outcome expression for a direct call
    result, propagated only through Boolean operations that preserve an affine
    GF(2) expression.  Keeping the invocation identity separate prevents two
    Boolean results (or two calls to the same gadget) from being mistaken for
    the selected profile's success predicate.
    """

    definition: "GadgetDefinition | ProtocolDefinition"
    attempt: str
    analysis: "GadgetProfile | None"
    profile: str | None
    invocation: int
    parity: ProfileParity | None
    outcome_rows: tuple[ProfileParity, ...] = ()
    outcome_roles: tuple[tuple[str, ...], ...] = ()
    outcome_index: int | None = None
    all_false_rows: tuple[ProfileParity, ...] | None = None
    all_false_indices: tuple[int, ...] | None = None
    patch_results: tuple[Any, ...] = ()
    probability_source: str | None = None
    success_probability: float | None = None
    probability_evidence: str | None = None

    def same_invocation(self, other: object) -> bool:
        return (isinstance(other, _PredicateProvenance) and
                self.definition is other.definition and
                self.attempt == other.attempt and
                self.analysis is other.analysis and
                self.profile == other.profile and
                self.invocation == other.invocation)


@dataclass(frozen=True, slots=True)
class ParameterMap:
    """Bijection between realization and objective parameter names."""

    pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        pairs = self.pairs
        if isinstance(pairs, Mapping):
            pairs = tuple(pairs.items())
        pairs = tuple(tuple(item) for item in pairs)
        if any(
                len(item) != 2 or any(not isinstance(name, str) or not name
                                      for name in item)
                for item in pairs):
            raise ValueError(
                "ParameterMap pairs must be (realization, objective) name pairs"
            )
        realization = tuple(item[0] for item in pairs)
        objective = tuple(item[1] for item in pairs)
        if len(set(realization)) != len(realization):
            raise ValueError(
                "ParameterMap is not a bijection: a realization parameter "
                "name appears more than once")
        if len(set(objective)) != len(objective):
            raise ValueError(
                "ParameterMap is not a bijection: an objective parameter "
                "name appears more than once")
        object.__setattr__(self, "pairs", pairs)

    @property
    def realization_to_objective(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.pairs))

    @property
    def objective_to_realization(self) -> Mapping[str, str]:
        return MappingProxyType({
            objective: realization for realization, objective in self.pairs
        })
