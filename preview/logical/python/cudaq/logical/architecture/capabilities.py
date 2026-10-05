# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Typed physical-carrier capabilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PhysicalCapability:
    """One typed property that may be installed on physical carriers.

    ``key`` is only the stable serialization identity. Python programs pass
    capability values (normally a domain-specific subclass), never raw keys.
    """

    key: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("physical capability key must be nonempty")


# Canonical P3 capability for the variadic joint operation represented by
# ``phys.rotate_product``.  This is deliberately not a ``PhysicalAction``:
# product-rotation arity is the nonempty Pauli support selected by each event,
# whereas physical actions have a fixed positive arity (or unary broadcast
# semantics).
NATIVE_PAULI_PRODUCT_ROTATION = PhysicalCapability(
    "qlx.physical/native_pauli_product_rotation")


class HeraldedErasure(PhysicalCapability):
    """Carrier equipment that exposes located erasure observations."""

    def __init__(self) -> None:
        super().__init__("qlx.physical/heralded_erasure")


@dataclass(frozen=True, slots=True)
class PhysicalCapabilityBinding:
    """Install one typed capability on a selected carrier subset."""

    capability: PhysicalCapability
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.capability, PhysicalCapability):
            raise TypeError(
                "physical capability binding requires a PhysicalCapability")
        indices = tuple(self.indices)
        if any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in indices):
            raise TypeError("physical capability indices must be Python ints")
        if len(set(indices)) != len(indices):
            raise ValueError(
                "physical capability indices must not contain duplicates")
        object.__setattr__(self, "indices", tuple(sorted(indices)))


__all__ = [
    "HeraldedErasure",
    "NATIVE_PAULI_PRODUCT_ROTATION",
    "PhysicalCapability",
    "PhysicalCapabilityBinding",
]
