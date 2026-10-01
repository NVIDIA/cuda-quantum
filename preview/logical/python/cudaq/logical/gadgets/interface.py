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


class OutcomeRole(str, Enum):
    """Application role of one Boolean gadget outcome."""

    RESULT = "result"
    SUCCESS = "success"

    @classmethod
    def from_value(cls, value) -> "OutcomeRole":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError as exc:
                raise ValueError(f"unknown outcome role {value!r}") from exc
        raise TypeError("outcome roles must be OutcomeRole values")


def _freeze_gadget_metadata(value, *, what):
    """Detach public gadget metadata into a deterministic immutable tree."""

    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{what} keys must be nonempty strings")
            frozen[key] = _freeze_gadget_metadata(item, what=f"{what}.{key}")
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_gadget_metadata(item, what=what) for item in value)
    if isinstance(value, (set, frozenset)):
        raise TypeError(f"{what} sets are not supported")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{what} floating-point values must be finite")
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(
        f"{what} values must be scalar, mapping, list, or tuple values")


class patch(Generic[EncodingT]):
    """Annotation for one linear encoded patch boundary value."""


@dataclass(frozen=True, slots=True, eq=False)
class BlockEndpoint:
    """One typed encoded-block endpoint on a gadget interface.

    ``name`` is a diagnostic alias inherited from the Python signature.  The
    endpoint's identity is the gadget, side, and ordinal; profile authors never
    repeat the alias as a string.
    """

    gadget: "GadgetDefinition"
    side: Literal["input", "output"]
    index: int
    name: str
    encoding: Any
    code_profile: Any
    state: str = "initialized"
    ownership: str = "borrow"

    def __post_init__(self) -> None:
        if self.side not in {"input", "output"}:
            raise ValueError("block endpoint side must be input or output")
        if not isinstance(self.index, int) or isinstance(
                self.index, bool) or self.index < 0:
            raise TypeError("block endpoint index must be a nonnegative int")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("block endpoint diagnostic name must be nonempty")

    def __hash__(self) -> int:
        return hash((id(self.gadget), self.side, self.index))

    def __eq__(self, other) -> bool:
        return (isinstance(other, BlockEndpoint) and
                self.gadget is other.gadget and self.side == other.side and
                self.index == other.index)

    @property
    def syndrome(self) -> "SyndromeBundleRef":
        from .records import SyndromeBundleRef

        return SyndromeBundleRef(self)

    def __repr__(self) -> str:
        return f"{self.gadget.name}.{self.side}s.{self.name}"


class EndpointCollection:
    """Ordered, named view of one side of a gadget interface."""

    __slots__ = ("_endpoints", "_by_name")

    def __init__(self, endpoints: Iterable[BlockEndpoint]) -> None:
        self._endpoints = tuple(endpoints)
        self._by_name = MappingProxyType(
            {endpoint.name: endpoint for endpoint in self._endpoints})
        if len(self._by_name) != len(self._endpoints):
            raise ValueError(
                "gadget endpoint diagnostic names must be unique per side")

    @property
    def blocks(self) -> tuple[BlockEndpoint, ...]:
        return self._endpoints

    def only(self) -> BlockEndpoint:
        if len(self._endpoints) != 1:
            raise ValueError(
                f"expected exactly one block endpoint, found {len(self._endpoints)}"
            )
        return self._endpoints[0]

    def __iter__(self):
        return iter(self._endpoints)

    def __len__(self) -> int:
        return len(self._endpoints)

    def __getitem__(self, index: int) -> BlockEndpoint:
        return self._endpoints[index]

    def __getattr__(self, name: str) -> BlockEndpoint:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True, slots=True)
class BlockFlow:
    """A typed hyperedge relating encoded input and output endpoints."""

    inputs: tuple[BlockEndpoint, ...]
    outputs: tuple[BlockEndpoint, ...]
    kind: str
    pairs: tuple[tuple[BlockEndpoint, BlockEndpoint], ...] = ()
    transform: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "pairs", tuple(self.pairs))
        if not self.inputs and not self.outputs:
            raise ValueError(
                "a block flow requires an input or output endpoint")


@dataclass(frozen=True, slots=True)
class GadgetInterface:
    """Typed block-and-record boundary inferred from a gadget signature."""

    gadget: "GadgetDefinition"
    inputs: EndpointCollection
    outputs: EndpointCollection
    flows: tuple[BlockFlow, ...]

    @property
    def records(self) -> "GadgetRecords":
        from .records import GadgetRecords

        return GadgetRecords(self.gadget)
