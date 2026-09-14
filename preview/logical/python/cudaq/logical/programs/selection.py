# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class SelectionIntent:
    """Portable statistical intent over a program's logical results."""

    mode: str
    predicate: Callable[[Any], Any]

    def __post_init__(self) -> None:
        if self.mode not in {"require", "condition_results", "abort_on"}:
            raise ValueError(
                "selection mode must be require, condition_results, or abort_on"
            )
        if not callable(self.predicate):
            raise TypeError("selection predicate must be callable")

    @property
    def accept_when(self) -> bool:
        return self.mode != "abort_on"


def require(predicate: Callable[[Any], Any]) -> SelectionIntent:
    return SelectionIntent("require", predicate)


def condition_results(predicate: Callable[[Any], Any]) -> SelectionIntent:
    return SelectionIntent("condition_results", predicate)


def abort_on(predicate: Callable[[Any], Any]) -> SelectionIntent:
    return SelectionIntent("abort_on", predicate)


__all__ = [
    "SelectionIntent",
    "require",
    "condition_results",
    "abort_on",
]
