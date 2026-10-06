# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Typed logical gate sets and their required legalization pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import math


def _validate_precision(precision) -> float:
    if (not isinstance(precision,
                       (int, float)) or isinstance(precision, bool) or
            not math.isfinite(float(precision)) or
            not 0.0 < float(precision) < 1.0):
        raise ValueError("synthesis precision must be finite and in (0, 1)")
    return float(precision)


@dataclass(frozen=True, slots=True)
class GateSet:
    """One typed logical basis and the passes that establish it.

    Gate sets own legalization policy;
    :func:`cudaq.logical.compiler.synthesize` only executes
    the immutable pipeline supplied by the selected value.
    """

    name: str
    actions: tuple[str, ...]
    legalization_passes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("gate-set name must be nonempty")
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "legalization_passes",
                           tuple(self.legalization_passes))
        if not self.actions or not self.legalization_passes:
            raise ValueError(
                "a gate set requires actions and legalization passes")

    def pipeline(self, *, precision: float = 1.0e-10):
        """Return the P0-to-P0 pipeline required by this gate set."""

        from .pipeline import PassSpec, Pipeline

        precision = _validate_precision(precision)
        passes = []
        for name in self.legalization_passes:
            options = ()
            if name == "qlx-synthesize-rotations":
                options = (
                    ("gate_set", self.name),
                    ("precision", precision),
                )
            passes.append(PassSpec(name, options))
        return Pipeline(tuple(passes), output_profile="p0")


clifford_t = GateSet(
    name="clifford_t",
    actions=("h", "s", "t", "cx"),
    legalization_passes=(
        "qlx-synthesize-rotations",
        "qlx-verify-clifford-t",
    ),
)


def _match_pipeline(pipeline):
    """Return ``(gate_set, precision)`` for a canonical gate-set pipeline."""

    if not hasattr(pipeline, "passes"):
        return None
    for gate_set in (clifford_t,):
        if tuple(item.name for item in pipeline.passes) != (
                gate_set.legalization_passes):
            continue
        options = dict(pipeline.passes[0].options)
        if options.get("gate_set") != gate_set.name:
            continue
        precision = _validate_precision(options.get("precision"))
        if pipeline != gate_set.pipeline(precision=precision):
            continue
        return gate_set, precision
    return None


__all__ = ["GateSet", "clifford_t"]
