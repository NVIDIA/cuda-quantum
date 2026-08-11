# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Frozen benchmark corpus for GridSynth tuning.

The corpus is deliberately fixed: defaults are selected on the ``tuning``
split and validated on the ``heldout`` split, so the split must not move
between runs or the validation stops meaning anything. Angles are emitted as
decimal strings at high precision rather than floats, because gridsynth
accepts strings for arbitrary-precision input and a float would silently cap
the achievable accuracy at deep epsilon.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction

# Enough digits to stay well below the tightest epsilon in the corpus.
getcontext().prec = 120

# High-precision pi (120 significant digits).
PI = Decimal(
    "3.14159265358979323846264338327950288419716939937510582097494459230781"
    "6406286208998628034825342117067982148086513282306647093844")

# Seed for the random angle family. Fixed so the corpus is reproducible; it
# has nothing to do with the solver's internal RNG seed.
CORPUS_SEED = 20260810


@dataclass(frozen=True)
class Angle:
    """One benchmark angle.

    Attributes:
        name: Stable identifier, unique across the corpus.
        family: Which structural family the angle belongs to. Results are
            reported per family because the families stress different parts
            of the search.
        theta: Decimal string passed verbatim to the implementations.
        split: ``"tuning"`` or ``"heldout"``.
    """

    name: str
    family: str
    theta: str
    split: str


def _dec(value: Decimal) -> str:
    """Render a Decimal as a plain (non-exponential) decimal string."""
    return format(value.normalize(), "f")


def _dyadic() -> list[tuple[str, str]]:
    """pi / 2^n -- the angles QFT and Shor circuits are actually built from."""
    return [(f"pi_over_2^{n}", _dec(PI / (2**n))) for n in range(1, 13)]


def _rational() -> list[tuple[str, str]]:
    """pi * a / b for small coprime a/b. Generic 'hard' angles."""
    out = []
    for b in (3, 5, 7, 11, 13, 17, 19, 23, 29, 53):
        for a in (1, 2):
            if Fraction(a, b).denominator != b:
                continue
            out.append((f"pi_{a}_over_{b}", _dec(PI * a / b)))
    return out


def _random_angles(count: int = 24) -> list[tuple[str, str]]:
    """Uniform on (0, 2*pi) from a fixed seed."""
    rng = random.Random(CORPUS_SEED)
    out = []
    for i in range(count):
        # Draw 30 decimal digits of mantissa so the angle is not a float in
        # disguise once it reaches the arbitrary-precision solver.
        frac = Decimal(rng.getrandbits(100)) / Decimal(2**100)
        out.append((f"uniform_{i:02d}", _dec(2 * PI * frac)))
    return out


def _near_clifford() -> list[tuple[str, str]]:
    """m*pi/2 + delta for tiny delta.

    Exercises the zero-T shortcut boundary in gridsynth_unitary: just inside
    it the answer is a Clifford and synthesis is trivial, just outside it the
    full search runs. Mis-tuning that boundary shows up here first.
    """
    out = []
    for m in range(4):
        for exp in (3, 6, 9, 12):
            delta = Decimal(1).scaleb(-exp)
            out.append(
                (f"near_clifford_m{m}_1e-{exp}", _dec(PI * m / 2 + delta)))
    return out


def _adversarial() -> list[tuple[str, str]]:
    """Angles measured to hit the runtime tail.

    5*pi/53 costs ~50x its neighbours at eps=1e-30, reproducibly and
    independently of the timeout settings. Pinned here so the tail case is
    never accidentally dropped from the corpus.
    """
    return [
        ("tail_5pi_over_53", _dec(PI * 5 / 53)),
        ("tail_7pi_over_53", _dec(PI * 7 / 53)),
    ]


def _split_for(family: str, index: int) -> str:
    """Assign a split deterministically.

    Alternating by index keeps both splits structurally comparable -- each
    family contributes to both, so the held-out set is not a different kind
    of problem from the tuning set. The adversarial family goes entirely to
    tuning: those angles were found by looking at the data, so treating them
    as held-out evidence would be circular.
    """
    if family == "adversarial":
        return "tuning"
    return "tuning" if index % 2 == 0 else "heldout"


def angles() -> list[Angle]:
    """The full frozen angle corpus."""
    families = {
        "dyadic": _dyadic(),
        "rational": _rational(),
        "uniform": _random_angles(),
        "near_clifford": _near_clifford(),
        "adversarial": _adversarial(),
    }
    out: list[Angle] = []
    for family, entries in families.items():
        for index, (name, theta) in enumerate(entries):
            out.append(
                Angle(name=name,
                      family=family,
                      theta=theta,
                      split=_split_for(family, index)))
    return out


# Log-spaced tolerances. The interesting structure is above 1e-30: runtime
# steps sharply near 1e-35 while T-count keeps growing smoothly, so the grid
# is deliberately dense across that transition.
EPSILONS = [f"1e-{d}" for d in (4, 6, 8, 10, 15, 20, 25, 30, 33, 35, 37, 40)]

# Kept out of the default grid: a single call at these tolerances can take
# tens of seconds to many minutes at the current defaults. Opt in explicitly.
DEEP_EPSILONS = [f"1e-{d}" for d in (45, 50, 60)]


def summary() -> str:
    """One-line-per-family description of the corpus, for logs."""
    rows = {}
    for angle in angles():
        key = (angle.family, angle.split)
        rows[key] = rows.get(key, 0) + 1
    lines = [f"epsilons: {len(EPSILONS)} ({EPSILONS[0]} .. {EPSILONS[-1]})"]
    for (family, split), count in sorted(rows.items()):
        lines.append(f"  {family:<14s} {split:<8s} {count:3d} angles")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
