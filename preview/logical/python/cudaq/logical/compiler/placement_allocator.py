# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Logical-space allocation state for P0-to-P1 placement."""

from __future__ import annotations

from collections import defaultdict

from ..architecture.logical import (
    LogicalMachine,
    LogicalValueGroup,
    LogicalValueRef,
    Space,
)
from ..architecture.constraints import (
    AllowSpaces,
    Prefer,
    RequireCapability,
)


class _Allocator:

    def __init__(self, machine: LogicalMachine, constraints) -> None:
        self.machine = machine
        self.constraints = tuple(constraints or ())
        allowed = None
        required = set()
        preferences: list[tuple[int, Prefer]] = []
        for constraint_index, constraint in enumerate(self.constraints):
            if isinstance(constraint, AllowSpaces):
                constraint_spaces = set(constraint.spaces)
                allowed = (constraint_spaces if allowed is None else allowed &
                           constraint_spaces)
            elif isinstance(constraint, RequireCapability):
                required.add(constraint.capability)
            elif isinstance(constraint,
                            Prefer) and constraint.space is not None:
                preferences.append((constraint_index, constraint))
        candidates = [
            space for space in machine.spaces
            if allowed is None or space in allowed
        ]
        candidates = [
            space for space in candidates
            if required.issubset(set(space.capabilities))
        ]
        if not candidates:
            raise ValueError(
                "no machine space satisfies the placement constraints")
        self.candidates = tuple(candidates)
        self.preferences = tuple(preferences)
        self.relaxed_preferences: list[str] = []
        self._relaxed_preference_set: set[str] = set()
        self.used_slots: dict[str, set[int]] = defaultdict(set)

    @staticmethod
    def _reference_key(reference):
        identity = (reference.allocation
                    if reference.allocation is not None else reference.group)
        return identity, reference.path[0]

    @classmethod
    def _preference_applies(cls, preference, value_keys) -> bool:
        target = preference.for_
        if target is None or type(target) in (str, bool, int, float):
            # Scalars describe workload-wide lifecycle or policy scopes.
            return True
        if isinstance(target, LogicalValueRef):
            return cls._reference_key(target) in value_keys
        if isinstance(target, LogicalValueGroup):
            identity = (target.allocation
                        if target.allocation is not None else target.name)
            return any(key_identity == identity and 0 <= index < target.count
                       for key_identity, index in value_keys)
        if isinstance(target, (tuple, list)):
            return any(
                cls._reference_key(reference) in value_keys
                for reference in target)
        return False

    def _candidates_for(self, value_keys):
        preferred_names = {}
        for _, preference in self.preferences:
            if self._preference_applies(preference, value_keys):
                preferred_names.setdefault(
                    preference.space.name,
                    len(preferred_names),
                )
        if not preferred_names:
            return self.candidates
        fallback = len(preferred_names)
        return tuple(
            sorted(
                self.candidates,
                key=lambda space: preferred_names.get(space.name, fallback),
            ))

    @staticmethod
    def _value_scope(value_key) -> str:
        identity, index = value_key
        prefix = "allocation" if isinstance(identity, int) else "group"
        return f"{prefix}:{identity}[{index}]"

    def _record_relaxations(self, selected, value_keys) -> None:
        for value_key in sorted(
                value_keys,
                key=lambda item: (str(item[0]), item[1]),
        ):
            applicable = tuple(
                (index, preference)
                for index, preference in self.preferences
                if self._preference_applies(preference, {value_key}))
            for constraint_index, preference in applicable:
                if preference.space.name == selected.name:
                    break
                relaxation = (f"preference[{constraint_index}]:"
                              f"space={preference.space.name}:"
                              f"for={self._value_scope(value_key)}")
                if relaxation in self._relaxed_preference_set:
                    continue
                self._relaxed_preference_set.add(relaxation)
                self.relaxed_preferences.append(relaxation)

    def take(
            self,
            *,
            space: Space | None = None,
            minimum_remaining: int = 1,
            slot: int | None = None,
            value_keys=(),
    ) -> tuple[Space, int]:
        candidates = (self._candidates_for(frozenset(value_keys))
                      if space is None else (space,))
        if space is not None and space not in self.candidates:
            raise ValueError(
                f"colocation selected space {space.name!r} outside placement constraints"
            )
        if slot is not None and space is None:
            raise ValueError(
                "an exact placement slot requires one explicit space")
        for candidate in candidates:
            occupied = self.used_slots[candidate.name]
            if slot is not None:
                if candidate.capacity is not None and slot >= candidate.capacity:
                    raise ValueError(
                        f"slot {slot} exceeds space {candidate.name!r} capacity"
                    )
                if slot in occupied:
                    raise ValueError(
                        f"slot {slot} in space {candidate.name!r} is already occupied"
                    )
                occupied.add(slot)
                self._record_relaxations(candidate, value_keys)
                return candidate, slot
            if candidate.capacity is not None:
                available = candidate.capacity - len(occupied)
                if available < minimum_remaining:
                    continue
                selected = next(index for index in range(candidate.capacity)
                                if index not in occupied)
            else:
                selected = 0
                while selected in occupied:
                    selected += 1
            occupied.add(selected)
            self._record_relaxations(candidate, value_keys)
            return candidate, selected
        raise ValueError("machine logical capacity is exhausted")


__all__ = ["_Allocator"]
