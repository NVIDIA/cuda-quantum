# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class LoweringSpec:
    """One target capability recipe: cloned stages followed by finalization."""

    stages: tuple
    finalize: Callable
    accepted_stages: tuple[str, ...] = ()
    required_facets: tuple[str, ...] = ()
    produced_stage: str | None = None
    provides_facets: tuple[str, ...] = ()
    result_schema: str | None = None
    effect: str = "local"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(
            self,
            "accepted_stages",
            tuple(
                str(getattr(value, "value", value))
                for value in self.accepted_stages),
        )
        object.__setattr__(
            self,
            "required_facets",
            tuple(
                str(getattr(value, "value", value))
                for value in self.required_facets),
        )
        if self.produced_stage is not None:
            object.__setattr__(
                self,
                "produced_stage",
                str(getattr(self.produced_stage, "value", self.produced_stage)),
            )
        object.__setattr__(
            self,
            "provides_facets",
            tuple(
                str(getattr(value, "value", value))
                for value in self.provides_facets),
        )
        if self.effect not in {"local", "filesystem", "external"}:
            raise ValueError(
                "lowering effect must be local, filesystem, or external")
