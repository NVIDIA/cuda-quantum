# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Typed references used to bind ideal operands to encoded logical ports."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Iterator

from cudaq.logical._core.compat import UNSET, resolve_alias


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ActionOperandRef:
    """One named operand of an ideal logical action."""

    action: Any
    name: str
    index: int

    def __init__(self,
                 action=UNSET,
                 name=UNSET,
                 index=UNSET,
                 *,
                 objective=UNSET):
        owner = resolve_alias(action,
                              objective,
                              name="action",
                              legacy_name="objective")
        if name is UNSET or index is UNSET:
            raise TypeError("ActionOperandRef requires name and index")
        object.__setattr__(self, "action", owner)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "index", index)

    @property
    def objective(self):
        """Compatibility alias for the owning logical action."""

        return self.action

    def __hash__(self) -> int:
        return hash((id(self.action), self.name, self.index))

    def __eq__(self, other) -> bool:
        return (isinstance(other, ActionOperandRef) and
                self.action is other.action and self.name == other.name and
                self.index == other.index)

    def __repr__(self) -> str:
        owner = getattr(self.action, "name", type(self.action).__name__)
        return f"{owner}.operands.{self.name}"


@dataclass(frozen=True, slots=True, eq=False)
class LogicalPortRef:
    """One protected logical degree exposed by an encoding."""

    encoding: Any
    name: str
    index: int

    def __hash__(self) -> int:
        return hash((id(self.encoding), self.name, self.index))

    def __eq__(self, other) -> bool:
        return (isinstance(other, LogicalPortRef) and
                self.encoding is other.encoding and self.name == other.name and
                self.index == other.index)

    def __repr__(self) -> str:
        owner = getattr(self.encoding, "name", type(self.encoding).__name__)
        return f"{owner}.ports.{self.name}"


class _ReferenceNamespace:
    """Ordered attribute/index view over immutable named references."""

    __slots__ = ("_owner", "_refs", "_by_name", "_by_index", "_kind")

    def __init__(self, owner, refs: Iterable[Any], *, kind: str) -> None:
        refs = tuple(refs)
        by_name = {ref.name: ref for ref in refs}
        by_index = {ref.index: ref for ref in refs}
        if len(by_name) != len(refs) or len(by_index) != len(refs):
            raise ValueError(f"{kind} names and indices must be unique")
        self._owner = owner
        self._refs = refs
        self._by_name = MappingProxyType(by_name)
        self._by_index = MappingProxyType(by_index)
        self._kind = kind

    def __iter__(self) -> Iterator[Any]:
        return iter(self._refs)

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, key: str | int):
        if isinstance(key, bool) or not isinstance(key, (str, int)):
            raise TypeError(
                f"{self._kind} lookup expects a name or integer index")
        table = self._by_name if isinstance(key, str) else self._by_index
        try:
            return table[key]
        except KeyError as exc:
            raise KeyError(f"unknown {self._kind} {key!r}") from exc

    def __getattr__(self, name: str):
        try:
            return self._by_name[name]
        except KeyError as exc:
            owner = getattr(self._owner, "name", type(self._owner).__name__)
            raise AttributeError(
                f"{owner!r} has no {self._kind} named {name!r}") from exc

    def __dir__(self):
        return sorted((*super().__dir__(), *self._by_name))

    def __repr__(self) -> str:
        return f"{self._kind}s({', '.join(self._by_name)})"


class ActionOperands(_ReferenceNamespace):
    """Typed namespace returned by a logical action's ``operands``."""

    def __init__(self, action=UNSET, names=UNSET, *, objective=UNSET) -> None:
        owner = resolve_alias(action,
                              objective,
                              name="action",
                              legacy_name="objective")
        if names is UNSET:
            raise TypeError("ActionOperands requires operand names")
        refs = tuple(
            ActionOperandRef(owner, str(name), index)
            for index, name in enumerate(names))
        super().__init__(owner, refs, kind="action operand")


class LogicalPorts(_ReferenceNamespace):
    """Typed namespace returned by ``encoding.ports``."""

    def __init__(self, encoding) -> None:
        refs = tuple(
            LogicalPortRef(
                encoding,
                name,
                encoding.logical_port_indices[name],
            ) for name in encoding.logical_ports)
        super().__init__(encoding, refs, kind="logical port")


def standard_action_operand_names(name: str, arity: int) -> tuple[str, ...]:
    """Canonical operand names for built-in logical actions."""

    named = {
        "cx": ("control", "target"),
        "cz": ("left", "right"),
        "ccz": ("a", "b", "c"),
        "ccx": ("control_a", "control_b", "target"),
    }.get(name)
    return named or tuple(f"q{index}" for index in range(arity))


# Keep identity-based ownership and equality identical through legacy names.
ObjectiveOperandRef = ActionOperandRef
ObjectiveOperands = ActionOperands
standard_objective_operand_names = standard_action_operand_names

__all__ = [
    "LogicalPortRef",
    "LogicalPorts",
    "ActionOperandRef",
    "ActionOperands",
    "ObjectiveOperandRef",
    "ObjectiveOperands",
]
