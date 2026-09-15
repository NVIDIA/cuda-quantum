# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Reusable folded-round policies for generated QEC protocols."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoundPolicy:
    name: str

    def resolve(self, code, policy=None) -> int:
        policy = dict(policy or {})
        if self.name == "from_code_distance":
            override = policy.get("rounds")
            if override is not None:
                if not isinstance(override, int) or isinstance(
                        override, bool) or override <= 0:
                    raise TypeError(
                        "QEC rounds override must be a positive int")
                return override
            distance = getattr(code.d, "value", None)
            if distance is None:
                raise ValueError(
                    "rounds-from-distance requires distance evidence or policy rounds="
                )
            return int(distance)
        raise ValueError(f"unknown QEC round policy {self.name!r}")


from_code_distance = RoundPolicy("from_code_distance")

__all__ = ["RoundPolicy", "from_code_distance"]
