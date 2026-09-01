# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Concrete protocol definitions and their closed-form metadata."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .. import std
from .distillation import (
    FIFTEEN_TO_ONE_ROTATION_STEPS,
    FIFTEEN_TO_ONE_ROTATION_SUPPORTS,
    bare_measure_x,
    bare_s,
    distill_15to1,
)


@dataclass(frozen=True, slots=True, init=False)
class _FifteenToOneProductionModel:
    """Fixed closed-form model for the retained 15-to-1 T factory.

    The exact probability equations below are protocol-specific.  Keeping the
    model non-public and non-configurable prevents arbitrary protocol metadata
    from being silently evaluated with 15-to-1 mathematics.
    """

    name: str = "distill-15to1-T"
    produces: std.ResourceKind = std.T_STATE
    raw_inputs: int = 15
    scratch_blocks: int = 11
    cycles_per_attempt: int = 120
    pipeline_depth: int = 4
    error_coefficient: float = 35.0
    error_order: int = 3
    rejection_coefficient: float = 15.0

    @staticmethod
    def _input_error(p: float) -> float:
        p = float(p)
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError("15-to-1 input error must be finite and in [0, 1]")
        return p

    def output_error(self, p: float) -> float:
        """Return the exact conditional output error for independent inputs."""

        p = self._input_error(p)
        if p > 0.5:
            # Complementing every independent input complements the accepted
            # logical output. Evaluate the small complementary probability to
            # keep the exact symmetry numerically stable near p=1.
            return 1.0 - self.output_error(1.0 - p)
        # The undetected-error weight enumerator is evaluated in p rather than
        # as a difference of powers of (1 - 2p). This Horner form retains the
        # exact polynomial while avoiding cancellation at product-scale p.
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


DISTILL_15TO1_T = _FifteenToOneProductionModel()

__all__ = [
    "DISTILL_15TO1_T",
    "FIFTEEN_TO_ONE_ROTATION_SUPPORTS",
    "FIFTEEN_TO_ONE_ROTATION_STEPS",
    "distill_15to1",
    "bare_measure_x",
    "bare_s",
]
