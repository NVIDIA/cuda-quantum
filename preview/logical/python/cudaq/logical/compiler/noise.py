# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================ #
"""Standard circuit noise shared by circuit export and finite fault analysis."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class StandardDepolarizingNoise:
    """Circuit depolarization with physical reset and measurement flips.

    Round-start data noise requires explicit instruction sites. Pauli channels
    retain categorical alternatives; no independent-Pauli approximation is made.
    """

    after_clifford_depolarization: float = 0.0
    before_round_data_depolarization: float = 0.0
    before_measure_flip_probability: float = 0.0
    after_reset_flip_probability: float = 0.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value,
                                                          (int, float)) or
                    not isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite in [0, 1]")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True, slots=True)
class _Channel:
    kind: str
    carriers: tuple[int, ...]
    probability: float


def _standard_noise_placement(projection,
                              noise,
                              round_starts=(),
                              data_carriers=None):
    """Return before/after channels indexed by projected instruction.

    After-Clifford channels follow the physical gate groups in each instruction.
    Both consumers use these placements; neither invents a second noise policy.
    """
    from .clifford import CliffordProjection

    if not isinstance(projection, CliffordProjection):
        raise TypeError("noise placement requires a CliffordProjection")
    if not isinstance(noise, StandardDepolarizingNoise):
        raise TypeError("noise must be StandardDepolarizingNoise")
    if isinstance(round_starts, str):
        raise ValueError("round starts must be distinct instruction site IDs")
    selected = tuple(round_starts)
    if (any(not isinstance(site, str) for site in selected) or
            len(set(selected)) != len(selected)):
        raise ValueError("round starts must be distinct instruction site IDs")
    if not set(selected) <= {op.site_id for op in projection.instructions}:
        raise ValueError("round start does not name a projected instruction")
    if data_carriers is None:
        boundaries = projection.interface.inputs or projection.interface.outputs
        data_carriers = tuple(
            dict.fromkeys(
                q for boundary in boundaries for q in boundary.data_carriers))
    else:
        data_carriers = tuple(data_carriers)
    if (len(set(data_carriers)) != len(data_carriers) or
            any(not isinstance(q, int) or isinstance(q, bool) or
                not 0 <= q < projection.interface.carrier_count
                for q in data_carriers)):
        raise ValueError("data carriers must be distinct projected carriers")
    before, after = [], []
    for op in projection.instructions:
        pre, post = [], []
        if op.site_id in selected and noise.before_round_data_depolarization:
            pre.extend(
                _Channel("DEPOLARIZE1", (
                    q,), noise.before_round_data_depolarization)
                for q in data_carriers)
        if op.kind == "MPP":
            if noise.before_measure_flip_probability:
                raise ValueError(
                    "measurement-flip noise on MPP requires an explicit fault model"
                )
        elif op.kind in ("M", "MX"):
            if noise.before_measure_flip_probability:
                pre.append(
                    _Channel("X_ERROR" if op.kind == "M" else "Z_ERROR",
                             op.carriers,
                             noise.before_measure_flip_probability))
        elif op.kind in ("R", "RX"):
            if noise.after_reset_flip_probability:
                post.append(
                    _Channel("X_ERROR" if op.kind == "R" else "Z_ERROR",
                             op.carriers, noise.after_reset_flip_probability))
        elif noise.after_clifford_depolarization:
            width = 2 if op.kind in ("CX", "CZ", "SWAP") else 1
            post.extend(
                _Channel("DEPOLARIZE2" if width == 2 else "DEPOLARIZE1", op.
                         carriers[i:i +
                                  width], noise.after_clifford_depolarization)
                for i in range(0, len(op.carriers), width))
        before.append(tuple(pre))
        after.append(tuple(post))
    return tuple(before), tuple(after)


__all__ = ["StandardDepolarizingNoise"]
