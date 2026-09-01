# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


class index(int):
    """CUDA-Q Logical compile/runtime index annotation."""


class float64(float):
    """CUDA-Q Logical 64-bit floating-point annotation."""


SchemaT = TypeVar("SchemaT")
KindT = TypeVar("KindT")
PayloadT = TypeVar("PayloadT")
DomainT = TypeVar("DomainT")


class logical_record(Generic[SchemaT]):
    pass


class logical_resource(Generic[KindT]):
    pass


class logical_event(Generic[PayloadT]):
    pass


class logical_frame(Generic[DomainT]):
    pass


class record(Generic[SchemaT]):
    pass


class resource(Generic[KindT]):
    pass


class event(Generic[PayloadT]):
    pass


class frame(Generic[DomainT]):
    pass


@dataclass(frozen=True, slots=True)
class LogicalState:
    """Portable preparation intent for one logical state.

    The value names an instrument objective; it does not embed a circuit,
    encoding, or device realization. Those remain linked P2 definitions.
    """

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("logical state name must be a nonempty string")


zero = LogicalState("zero")
plus = LogicalState("plus")
