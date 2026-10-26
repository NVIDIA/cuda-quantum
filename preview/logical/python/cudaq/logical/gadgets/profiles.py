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

from .interface import (
    BlockEndpoint,
    GadgetInterface,
    OutcomeRole,
    _freeze_gadget_metadata,
)
from .records import (
    InputSyndromeRef,
    ProfileBinding,
    ProfileParity,
    ProfileVectorExpr,
    RecordRef,
    RecordVectorParity,
    SyndromeBundleRef,
)
from .semantics import (
    OutputSyndromeAssignment,
    SuccessPredicate,
    _reconcile_profile_role,
    _scalar_profile_parities,
)
from .specification import _gadget_boundary_profiles


@dataclass(frozen=True, slots=True)
class GadgetProfile:
    """Immutable success and boundary analysis for one unchanged gadget."""

    gadget: "GadgetDefinition"
    code_profile: Any | None = None
    input_profiles: Mapping[BlockEndpoint, Any] | None = None
    output_profiles: Mapping[BlockEndpoint, Any] | None = None
    success: tuple[SuccessPredicate, ...] = ()
    boundary: Mapping[SyndromeBundleRef, Any] | None = None
    output_syndromes: tuple[OutputSyndromeAssignment, ...] = ()
    boundary_complete: bool = False
    name: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        # Local import closes the lifecycle cycle: definitions own profiles,
        # while profile instances validate the definition they describe.
        from .definition import GadgetDefinition

        if not isinstance(self.gadget, GadgetDefinition):
            raise TypeError(
                "GadgetProfile requires a @cudaq.logical.gadget definition")
        from cudaq.logical.codes import CodeProfile

        inferred_inputs, inferred_outputs = _gadget_boundary_profiles(
            self.gadget)
        if self.code_profile is not None:
            if not isinstance(self.code_profile, CodeProfile):
                raise TypeError(
                    "GadgetProfile code_profile must be cudaq.logical.CodeProfile"
                )
            for label, inferred in (
                ("input", inferred_inputs),
                ("output", inferred_outputs),
            ):
                if any(profile is not self.code_profile
                       for profile in inferred.values()):
                    raise ValueError(
                        f"GadgetProfile code_profile is not the profile carried "
                        f"by every typed {label} endpoint Encoding")

        def normalize_profiles(provided, inferred, label):
            values = dict(inferred if provided is None else provided)
            if set(values) != set(inferred):
                raise ValueError(
                    f"GadgetProfile {label}_profiles must cover exactly "
                    f"{tuple(inferred)}")
            for endpoint, profile in values.items():
                if not isinstance(endpoint, BlockEndpoint):
                    raise TypeError(
                        f"GadgetProfile {label}_profiles keys must be BlockEndpoint values"
                    )
                if not isinstance(profile, CodeProfile):
                    raise TypeError(
                        f"GadgetProfile {label} endpoint {endpoint!r} must use CodeProfile"
                    )
                if profile is not inferred[endpoint]:
                    raise ValueError(
                        f"GadgetProfile {label} endpoint {endpoint!r} profile "
                        "is not carried by that endpoint Encoding")
            return MappingProxyType(values)

        input_profiles = normalize_profiles(self.input_profiles,
                                            inferred_inputs, "input")
        output_profiles = normalize_profiles(self.output_profiles,
                                             inferred_outputs, "output")
        object.__setattr__(self, "input_profiles", input_profiles)
        object.__setattr__(self, "output_profiles", output_profiles)
        all_profiles = tuple(
            (*input_profiles.values(), *output_profiles.values()))
        if self.code_profile is None and all_profiles and all(
                profile is all_profiles[0] for profile in all_profiles):
            object.__setattr__(self, "code_profile", all_profiles[0])
        success = tuple(self.success)
        if any(not isinstance(predicate, SuccessPredicate)
               for predicate in success):
            raise TypeError(
                "GadgetProfile success must contain SuccessPredicate values")
        profile_name = self.name or f"{self.gadget.name}_profile"
        success = tuple(
            SuccessPredicate(parity) for parity in _reconcile_profile_role(
                self.gadget,
                profile_name,
                OutcomeRole.SUCCESS,
                _scalar_profile_parities(success),
            ))
        if self.boundary is not None and self.output_syndromes:
            raise ValueError(
                "specify vector boundary bindings, not boundary plus normalized rows"
            )
        normalized_boundary = {}
        assignments = []
        if self.boundary is not None:
            for target, value in self.boundary.items():
                if not isinstance(target, SyndromeBundleRef):
                    raise TypeError(
                        "GadgetProfile boundary keys must be endpoint.syndrome bundles"
                    )
                endpoint = target.endpoint
                profile = output_profiles.get(endpoint)
                if profile is None:
                    raise ValueError(
                        "GadgetProfile boundary target is not an output of its gadget"
                    )
                expression = ProfileVectorExpr.from_value(
                    value, width=profile.effective_stabilizers.nrows)
                normalized_boundary[target] = expression
                assignments.extend(
                    OutputSyndromeAssignment(endpoint, index, row)
                    for index, row in enumerate(expression.rows))
        else:
            assignments.extend(self.output_syndromes)
            grouped = {}
            for assignment in assignments:
                if isinstance(assignment, OutputSyndromeAssignment):
                    grouped.setdefault(assignment.endpoint,
                                       []).append(assignment)
            for endpoint, rows in grouped.items():
                ordered = tuple(
                    item.parity
                    for item in sorted(rows, key=lambda item: item.index))
                normalized_boundary[endpoint.syndrome] = ProfileVectorExpr(
                    ordered)
        assignments = tuple(assignments)
        if any(not isinstance(assignment, OutputSyndromeAssignment)
               for assignment in assignments):
            raise TypeError("GadgetProfile output_syndromes must contain "
                            "OutputSyndromeAssignment values")
        object.__setattr__(self, "output_syndromes", assignments)
        object.__setattr__(self, "boundary",
                           MappingProxyType(normalized_boundary))
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "name", profile_name)
        object.__setattr__(
            self, "metadata",
            _freeze_gadget_metadata(dict(self.metadata or {}),
                                    what="GadgetProfile.metadata"))
        for expression in self.success:
            if isinstance(expression.parity, RecordVectorParity):
                parities = tuple(
                    ProfileParity(records=row.records)
                    for row in expression.parity.rows())
            elif isinstance(expression.parity, ProfileVectorExpr):
                parities = expression.parity.rows
            else:
                parities = (ProfileParity.from_value(expression.parity),)
            for parity in parities:
                if any(record.gadget is not self.gadget
                       for record in parity.records):
                    raise ValueError(
                        "a gadget profile may only reference records from its gadget"
                    )
                for syndrome in parity.input_syndromes:
                    profile = input_profiles.get(syndrome.endpoint)
                    if profile is None or syndrome.index >= (
                            profile.effective_stabilizers.nrows):
                        raise ValueError(
                            "profile parity references an invalid input syndrome"
                        )

        # A role-tagged OutcomeMap is the single algebraic authority for
        # success rows. This replay keeps the invariant local and explicit.
        _reconcile_profile_role(
            self.gadget,
            self.name,
            OutcomeRole.SUCCESS,
            _scalar_profile_parities(self.success),
        )

        seen_assignments = set()
        for assignment in assignments:
            profile = output_profiles.get(assignment.endpoint)
            if profile is None or assignment.index >= (
                    profile.effective_stabilizers.nrows):
                raise ValueError(
                    "output syndrome assignment has an invalid target")
            key = (assignment.endpoint, assignment.index)
            if key in seen_assignments:
                raise ValueError(
                    "output syndrome components must be assigned once")
            seen_assignments.add(key)
            for record in assignment.parity.records:
                if record.gadget is not self.gadget:
                    raise ValueError(
                        "output syndrome assignment references another gadget")
            for syndrome in assignment.parity.input_syndromes:
                input_profile = input_profiles.get(syndrome.endpoint)
                if input_profile is None or syndrome.index >= (
                        input_profile.effective_stabilizers.nrows):
                    raise ValueError(
                        "output syndrome assignment references an invalid input"
                    )
        expected_assignments = {
            (endpoint, index)
            for endpoint, profile in output_profiles.items()
            for index in range(profile.effective_stabilizers.nrows)
        }
        complete = seen_assignments == expected_assignments
        if self.boundary_complete and not complete:
            missing = tuple((endpoint.name, index)
                            for endpoint, index in sorted(
                                expected_assignments - seen_assignments,
                                key=lambda item: (item[0].index, item[1]),
                            ))
            raise ValueError(
                f"boundary-complete GadgetProfile is missing {missing}")
        object.__setattr__(self, "boundary_complete", complete)

    def _effective_role_parities(self, role):
        """Return the canonical affine table for success outcomes."""

        role = OutcomeRole.from_value(role)
        if role is not OutcomeRole.SUCCESS:
            raise ValueError("profile role must be success")
        return _reconcile_profile_role(
            self.gadget,
            self.name,
            role,
            _scalar_profile_parities(self.success),
        )

    @property
    def stage(self):
        from cudaq.logical.stages import P2

        return P2

    @property
    def facets(self):
        from cudaq.logical.stages import QEC_REALIZATION

        return (QEC_REALIZATION,)

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)


class ProfileGraph:
    """Advanced block-and-record graph builder for a detached profile.

    The graph is attached to a typed ``GadgetInterface``. ``bind`` adds one
    vector equation over a complete output syndrome bundle; scalar rows are
    introduced only when ``freeze`` normalizes the graph to ``GadgetProfile``.
    """

    __slots__ = (
        "interface",
        "name",
        "code_profile",
        "metadata",
        "_success",
        "_boundary",
    )

    def __init__(
        self,
        interface: GadgetInterface,
        *,
        code_profile=None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(interface, GadgetInterface):
            raise TypeError(
                "ProfileGraph requires a cudaq.logical.GadgetInterface")
        self.interface = interface
        self.name = name
        self.code_profile = code_profile
        self.metadata = metadata
        self._success = []
        self._boundary = {}

    def success(self, parity) -> "ProfileGraph":
        self._success.append(SuccessPredicate(parity))
        return self

    def bind(self, target: SyndromeBundleRef, value) -> "ProfileGraph":
        if not isinstance(target, SyndromeBundleRef):
            raise TypeError(
                "ProfileGraph.bind target must be endpoint.syndrome")
        if target.endpoint.gadget is not self.interface.gadget:
            raise ValueError("profile binding target belongs to another gadget")
        if target in self._boundary:
            raise ValueError("an output syndrome bundle may be bound only once")
        width = (self.code_profile.effective_stabilizers.nrows
                 if self.code_profile is not None else target.width)
        self._boundary[target] = ProfileVectorExpr.from_value(value,
                                                              width=width)
        return self

    @property
    def bindings(self) -> tuple[ProfileBinding, ...]:
        return tuple(
            ProfileBinding(target, value)
            for target, value in self._boundary.items())

    def freeze(self) -> GadgetProfile:
        return GadgetProfile(
            self.interface.gadget,
            code_profile=self.code_profile,
            success=tuple(self._success),
            boundary=self._boundary,
            name=self.name,
            metadata=self.metadata,
        )
