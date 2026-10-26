# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Typed names shared by logical and physical scheduling surfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SchedulingStrategy:
    """One typed scheduling choice in a specific semantic domain."""

    name: str
    domain: str
    compatible_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, what in ((self.name, "name"), (self.domain, "domain")):
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"scheduling strategy {what} must be a nonempty string")
        compatible_domains = tuple(self.compatible_domains)
        if any(not isinstance(value, str) or not value
               for value in compatible_domains):
            raise ValueError(
                "scheduling strategy compatible domains must be nonempty strings"
            )
        if self.domain in compatible_domains:
            raise ValueError(
                "a scheduling strategy's primary domain cannot be repeated as "
                "a compatible domain")
        if len(set(compatible_domains)) != len(compatible_domains):
            raise ValueError(
                "scheduling strategy compatible domains must be unique")
        object.__setattr__(self, "compatible_domains", compatible_domains)

    def __str__(self) -> str:
        return self.name

    @property
    def domains(self) -> tuple[str, ...]:
        """Every semantic domain in which this exact strategy is executable."""

        return (self.domain, *self.compatible_domains)

    def supports(self, domain: str) -> bool:
        """Whether the strategy is defined for ``domain``."""

        return domain in self.domains


class _Scheduling:
    minimize_makespan = SchedulingStrategy(
        "minimize_makespan",
        "physical",
    )
    minimize_active_volume = SchedulingStrategy(
        "minimize_active_volume",
        "physical",
    )
    greedy_asap = SchedulingStrategy(
        "greedy_asap",
        "lattice_surgery",
        ("physical",),
    )


scheduling = _Scheduling()

__all__ = ["SchedulingStrategy", "scheduling"]
