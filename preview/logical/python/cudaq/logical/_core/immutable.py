# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Shared sealing support for manually implemented immutable model values."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def _freeze_container(value: Any, active: set[int]) -> Any:
    """Defensively copy and recursively freeze standard mutable containers."""

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError("immutable model metadata cannot contain cycles")
        active.add(identity)
        try:
            return MappingProxyType({
                key: _freeze_container(item, active)
                for key, item in value.items()
            })
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError("immutable model metadata cannot contain cycles")
        active.add(identity)
        try:
            return tuple(_freeze_container(item, active) for item in value)
        finally:
            active.remove(identity)
    if isinstance(value, (set, frozenset)):
        identity = id(value)
        if identity in active:
            raise ValueError("immutable model metadata cannot contain cycles")
        active.add(identity)
        try:
            return frozenset(_freeze_container(item, active) for item in value)
        finally:
            active.remove(identity)
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def freeze_mapping(value: Mapping[Any, Any] | None) -> Mapping[Any, Any]:
    """Return an alias-independent, recursively immutable mapping."""

    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("immutable model metadata must be a mapping")
    return _freeze_container(value, set())


def freeze_value(value: Any) -> Any:
    """Return an alias-independent immutable form of a model field value."""

    return _freeze_container(value, set())


class ImmutableValue:
    """Reject public mutation after a model value finishes construction."""

    __slots__ = ("_immutable_sealed",)

    def _seal(self) -> None:
        object.__setattr__(self, "_immutable_sealed", True)

    def _is_sealed(self) -> bool:
        try:
            return object.__getattribute__(self, "_immutable_sealed")
        except AttributeError:
            return False

    def __setattr__(self, name, value) -> None:
        if self._is_sealed():
            raise AttributeError(
                f"qlx.{type(self).__name__} is immutable; cannot assign "
                f"{name!r}; construct a new value")
        object.__setattr__(self, name, value)

    def __delattr__(self, name) -> None:
        if self._is_sealed():
            raise AttributeError(
                f"qlx.{type(self).__name__} is immutable; cannot delete "
                f"{name!r}; construct a new value")
        object.__delattr__(self, name)
