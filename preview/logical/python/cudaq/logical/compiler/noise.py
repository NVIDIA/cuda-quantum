# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================ #
"""Standard circuit noise shared by circuit export and finite fault analysis."""

from dataclasses import dataclass
from math import isfinite


def _probability(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be finite in [0, 1]")
    return float(value)


@dataclass(frozen=True, slots=True)
class StandardDepolarizingNoise:
    """Circuit depolarization with physical reset and measurement flips.

    Channels attach only to physical operations. Additional noise layers use
    explicit ``Depolarize1Placement`` values. Pauli channels retain categorical
    alternatives; no independent-Pauli approximation is made.
    """

    after_clifford_depolarization: float = 0.0
    before_measure_flip_probability: float = 0.0
    after_reset_flip_probability: float = 0.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name,
                               _probability(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class Depolarize1Placement:
    """Independent one-qubit depolarization on explicit projected carriers.

    ``before`` is a native instruction site ID in the selected realization.
    Each carrier receives its own categorical X/Y/Z channel of total
    ``probability`` immediately before that instruction. Carrier indices are
    local to its projection. Circuit export and fault construction accept the
    same placements, independently of the operation-attached noise policy.
    """

    before: str
    carriers: tuple[int, ...]
    probability: float

    def __post_init__(self):
        if not isinstance(self.before, str) or not self.before:
            raise ValueError("before must be a nonempty instruction site ID")
        carriers = tuple(self.carriers)
        if (not carriers or
                any(not isinstance(q, int) or isinstance(q, bool) or q < 0
                    for q in carriers) or len(set(carriers)) != len(carriers)):
            raise ValueError("carriers must be distinct nonnegative integers")
        object.__setattr__(self, "carriers", carriers)
        object.__setattr__(self, "probability",
                           _probability(self.probability, "probability"))


@dataclass(frozen=True, slots=True)
class _Channel:
    kind: str
    carriers: tuple[int, ...]
    probability: float


def _standard_noise_placement(projection, noise, depolarizations=()):
    """Return before/after channels indexed by projected instruction.

    After-Clifford channels follow the physical gate groups in each instruction.
    Both consumers use these placements; neither invents a second noise policy.
    """
    from .clifford import CliffordProjection

    if not isinstance(projection, CliffordProjection):
        raise TypeError("noise placement requires a CliffordProjection")
    if not isinstance(noise, StandardDepolarizingNoise):
        raise TypeError("noise must be StandardDepolarizingNoise")
    sites = {op.site_id for op in projection.instructions}
    selected = {}
    for placement in depolarizations:
        if not isinstance(placement, Depolarize1Placement):
            raise TypeError(
                "depolarizations must contain Depolarize1Placement values")
        if placement.before not in sites:
            raise ValueError(
                "depolarization does not name a projected instruction")
        if any(q >= projection.interface.carrier_count
               for q in placement.carriers):
            raise ValueError("depolarization carrier is out of range")
        if placement.probability:
            selected.setdefault(placement.before, []).extend(
                _Channel("DEPOLARIZE1", (q,), placement.probability)
                for q in placement.carriers)
    before, after = [], []
    for op in projection.instructions:
        pre, post = list(selected.get(op.site_id, ())), []
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


__all__ = ["StandardDepolarizingNoise", "Depolarize1Placement"]
