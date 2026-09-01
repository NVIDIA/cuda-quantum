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

from ..programs.binding import (
    LogicalPortRef,
    ObjectiveOperandRef,
)
from .._core.immutable import ImmutableValue

EncodingT = TypeVar("EncodingT")

from .interface import BlockEndpoint, OutcomeRole, patch
from .records import ProfileParity, RecordRef


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
        from ..algebra.gf2 import (
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
            raise TypeError("OutcomeMap role must be a OutcomeRole value")
        if role not in {
                OutcomeRole.RESULT,
                OutcomeRole.SUCCESS,
        }:
            raise ValueError("OutcomeMap role must be result or success")
        return tuple(
            index for index, roles in enumerate(self.roles) if role in roles)


@dataclass(frozen=True, slots=True)
class _PredicateProvenance:
    """Typed affine meaning of one Boolean inside a protocol trace.

    ``parity`` is the exact GadgetSpec outcome expression for a direct call
    result, propagated only through Boolean operations that preserve an affine
    GF(2) expression.  Keeping the invocation identity separate prevents two
    Boolean results (or two calls to the same gadget) from being mistaken for
    an attempt's success predicate.
    """

    definition: "GadgetDefinition | ProtocolDefinition"
    attempt: str
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
