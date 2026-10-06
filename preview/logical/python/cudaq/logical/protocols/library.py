# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Ordinary QLX resource-protocol definitions and analytical models."""

from __future__ import annotations

from dataclasses import dataclass
import math

from cudaq.logical.ops._impl import (
    discard,
    produce,
    request_many,
)
from cudaq.logical.architecture.logical import Space
from cudaq.logical.protocols.definition import protocol
from cudaq.logical.types.semantic import resource
from .. import std

from .steane import (
    steane_logical_s,
    steane_t_injection,
    steane_teleportation_measurement,
    steane_teleportation_measurement_intent,
)
from .cultivation import (
    COLOR_3,
    COLOR_3_CARRIERS,
    COLOR_5,
    COLOR_5_CARRIERS,
    color_3,
    color_3_to_5,
    color_5,
    cultivate_color_3_to_5,
    grow_color_3_to_5,
    grow_color_3_to_5_profile,
)
from .actual_cultivation import (
    CULTIVATED_MATCHABLE_D6,
    cultivate_t_d3_to_matchable_d6,
    cultivated_matchable_d6,
    prepare_cultivated_t,
)

_T_STATE_FACTORY = Space(name="t_state_factory")
_CCZ_STATE_FACTORY = Space(name="ccz_state_factory")


@dataclass(frozen=True, slots=True)
class ProductionModel:
    name: str
    produces: std.ResourceKind
    raw_inputs: int
    scratch_blocks: int
    cycles_per_attempt: int
    pipeline_depth: int
    error_coefficient: float
    error_order: int
    rejection_coefficient: float

    @staticmethod
    def _input_error(p: float) -> float:
        p = float(p)
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError("factory input error must be finite and in [0, 1]")
        return p

    def output_error(self, p: float) -> float:
        p = self._input_error(p)
        return min(1.0, self.error_coefficient * p**self.error_order)

    def acceptance_probability(self, p: float) -> float:
        p = self._input_error(p)
        return max(0.0, min(1.0, 1.0 - self.rejection_coefficient * p))


class _FifteenToOneProductionModel(ProductionModel):
    """Exact analytical model for the retained 15-to-1 T factory."""

    def output_error(self, p: float) -> float:
        """Return the exact conditional output error for independent inputs."""

        p = self._input_error(p)
        if p > 0.5:
            return 1.0 - self.output_error(1.0 - p)
        polynomial = 1024.0
        for coefficient in (
                -7680.0,
                26880.0,
                -58240.0,
                87360.0,
                -96096.0,
                80080.0,
                -51360.0,
                25320.0,
                -9380.0,
                2478.0,
                -420.0,
                35.0,
        ):
            polynomial = polynomial * p + coefficient
        joint_error = p**3 * polynomial
        return joint_error / self.acceptance_probability(p)

    def acceptance_probability(self, p: float) -> float:
        """Return the exact acceptance probability for independent inputs."""

        p = self._input_error(p)
        polarization = 1.0 - 2.0 * p
        return (1.0 + 15.0 * polarization**8) / 16.0


DISTILL_15TO1_T = _FifteenToOneProductionModel(
    "distill-15to1-T",
    std.T_STATE,
    raw_inputs=15,
    scratch_blocks=11,
    cycles_per_attempt=120,
    pipeline_depth=4,
    error_coefficient=35.0,
    error_order=3,
    rejection_coefficient=15.0,
)

DISTILL_5TO1_T = ProductionModel(
    "distill-5to1-T",
    std.T_STATE,
    raw_inputs=5,
    scratch_blocks=5,
    cycles_per_attempt=80,
    pipeline_depth=2,
    error_coefficient=10.0,
    error_order=2,
    rejection_coefficient=5.0,
)


def per_unit_cell_error(distance: int, p: float) -> float:
    return 0.1 * (100.0 * float(p))**((int(distance) + 1) / 2.0)


def ccz_per_state_error(d1: int, d2: int, p: float) -> float:
    l2_topological = 1000.0 * per_unit_cell_error(d2, p)
    l1_distillation = 35.0 * (p + 100.0 * per_unit_cell_error(d1 // 2, p))**3
    l1_factory = 1100.0 * per_unit_cell_error(d1, p)
    return l2_topological + 28.0 * (l1_distillation + l1_factory)**2


@dataclass(frozen=True, slots=True)
class CCZProductionModel:
    d1: int = 15
    d2: int = 27
    cycles_per_attempt: int | None = None
    pipeline_depth: int = 1
    scratch_blocks: int = 0
    raw_inputs: int = 8
    name: str = "ccz-gidney-fowler"
    produces: std.ResourceKind = std.CCZ_STATE

    def __post_init__(self):
        if self.d1 <= 0 or self.d2 <= 0:
            raise ValueError("factory distances must be positive")

    @property
    def cycles(self):
        return self.cycles_per_attempt or 6 * self.d2

    def output_error(self, p: float) -> float:
        return ccz_per_state_error(self.d1, self.d2, p)

    def acceptance_probability(self, p: float) -> float:
        return max(0.0, min(1.0, 1.0 - 8.0 * float(p)))


def ccz_gidney_fowler_factory(*, d1=15, d2=27, **kwargs):
    return CCZProductionModel(d1=d1, d2=d2, **kwargs)


CCZ_GIDNEY_FOWLER = ccz_gidney_fowler_factory()

from .distillation import (
    CCZ_8TO1_CHECK_SUPPORTS,
    CCZ_8TO1_INJECTION_TARGETS,
    CCZ_8TO1_OUTPUT_CORRECTION_MASKS,
    CCZ_8TO1_SYNDROME_OUTPUTS,
    FIFTEEN_TO_ONE_ROTATION_STEPS,
    FIFTEEN_TO_ONE_ROTATION_SUPPORTS,
    bare_measure_x,
    bare_s,
    distill_15to1,
)


@protocol(
    implements=std.produce(std.T_STATE),
    metadata={
        "production_model": DISTILL_5TO1_T.name,
        "raw_inputs": DISTILL_5TO1_T.raw_inputs,
        "scratch_blocks": DISTILL_5TO1_T.scratch_blocks,
        "cycles_per_attempt": DISTILL_5TO1_T.cycles_per_attempt,
        "pipeline_depth": DISTILL_5TO1_T.pipeline_depth,
    },
)
def distill_5to1() -> resource[std.T_STATE]:
    raw = request_many(std.RAW_T_STATE, count=DISTILL_5TO1_T.raw_inputs)
    discard(raw)
    return produce(
        std.T_STATE,
        region=_T_STATE_FACTORY,
    )


@protocol(
    implements=std.produce(std.CCZ_STATE),
    metadata={
        "production_model": CCZ_GIDNEY_FOWLER.name,
        "d1": CCZ_GIDNEY_FOWLER.d1,
        "d2": CCZ_GIDNEY_FOWLER.d2,
        "raw_inputs": CCZ_GIDNEY_FOWLER.raw_inputs,
        "cycles_per_attempt": CCZ_GIDNEY_FOWLER.cycles,
    },
)
def ccz_gidney_fowler() -> resource[std.CCZ_STATE]:
    distilled_t = request_many(std.T_STATE, count=CCZ_GIDNEY_FOWLER.raw_inputs)
    discard(distilled_t)
    return produce(
        std.CCZ_STATE,
        region=_CCZ_STATE_FACTORY,
    )


__all__ = [
    "ProductionModel",
    "CCZProductionModel",
    "DISTILL_15TO1_T",
    "CCZ_8TO1_CHECK_SUPPORTS",
    "CCZ_8TO1_INJECTION_TARGETS",
    "CCZ_8TO1_OUTPUT_CORRECTION_MASKS",
    "CCZ_8TO1_SYNDROME_OUTPUTS",
    "FIFTEEN_TO_ONE_ROTATION_SUPPORTS",
    "FIFTEEN_TO_ONE_ROTATION_STEPS",
    "DISTILL_5TO1_T",
    "CCZ_GIDNEY_FOWLER",
    "per_unit_cell_error",
    "ccz_per_state_error",
    "ccz_gidney_fowler_factory",
    "distill_15to1",
    "bare_measure_x",
    "bare_s",
    "distill_5to1",
    "ccz_gidney_fowler",
    "steane_teleportation_measurement_intent",
    "steane_teleportation_measurement",
    "steane_logical_s",
    "steane_t_injection",
    "COLOR_3",
    "COLOR_5",
    "COLOR_3_CARRIERS",
    "COLOR_5_CARRIERS",
    "color_3",
    "color_5",
    "color_3_to_5",
    "cultivate_color_3_to_5",
    "grow_color_3_to_5",
    "grow_color_3_to_5_profile",
    "CULTIVATED_MATCHABLE_D6",
    "cultivate_t_d3_to_matchable_d6",
    "cultivated_matchable_d6",
    "prepare_cultivated_t",
]
