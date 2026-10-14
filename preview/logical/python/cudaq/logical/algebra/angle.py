# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Exact symbolic rotation angles: rational multiples of pi.

Authoring a rotation with :data:`cudaq.logical.pi` (and arithmetic on it) records the
angle as an exact rational multiple of pi rather than a lossy float.  This
makes the compiler's exactness guarantee *authoritative* instead of
*tolerance-inferred*: ``cudaq.logical.rz(q, cudaq.logical.pi/4)`` is
unambiguously the T-class
magic rotation, never a float that merely happens to land near ``pi/4``.

Under the canonical QLX convention ``R_P(theta) = exp(-i theta P / 2)`` the
quarter-turns of ``theta`` are the interesting points -- ``R_P(pi)`` is a
Pauli, ``R_P(pi/2)`` a Clifford, ``R_P(pi/4)`` the magic rotation -- so
``cudaq.logical.pi/4`` reads exactly as the intent (see the "Phase convention" section
of the lattice-surgery workflow docs).

The type stays deliberately thin: ``float(angle)`` yields the IEEE-nearest
radian value, so every existing float consumer keeps working unchanged, and
passing a raw Python float still takes the numeric/tolerance classification
path.  Both surfaces coexist.
"""

from __future__ import annotations

import math
from fractions import Fraction

__all__ = ["Angle", "pi"]


class Angle:
    """An angle equal to an exact rational multiple of pi: ``(num/den) * pi``.

    Instances are immutable and normalized (the fraction is reduced, the
    denominator kept positive).  Multiplication and division by integers or
    :class:`~fractions.Fraction` preserve exactness; mixing with a float
    falls back to a plain float, matching what a user would expect from
    ``cudaq.logical.pi * 0.5``.
    """

    __slots__ = ("_frac",)

    def __init__(self, num, den=1):
        # Fraction validates and reduces; a zero denominator raises here.
        self._frac = Fraction(num, den)

    @property
    def pi_fraction(self) -> tuple[int, int]:
        """The reduced ``(numerator, denominator)`` of the pi-coefficient."""
        return (self._frac.numerator, self._frac.denominator)

    def __float__(self) -> float:
        f = self._frac
        return f.numerator / f.denominator * math.pi

    # -- exactness-preserving arithmetic ------------------------------------
    def __mul__(self, other):
        if isinstance(other, (int, Fraction)) and not isinstance(other, bool):
            return Angle(self._frac * other)
        if isinstance(other, Angle):
            raise TypeError("cannot multiply two angles")
        return float(self) * other

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, Fraction)) and not isinstance(other, bool):
            return Angle(self._frac / other)
        return float(self) / other

    def __neg__(self):
        return Angle(-self._frac)

    def __pos__(self):
        return self

    def __add__(self, other):
        if isinstance(other, Angle):
            return Angle(self._frac + other._frac)
        return float(self) + other

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Angle):
            return Angle(self._frac - other._frac)
        return float(self) - other

    def __rsub__(self, other):
        if isinstance(other, Angle):
            return Angle(other._frac - self._frac)
        return other - float(self)

    # -- identity -----------------------------------------------------------
    def __eq__(self, other):
        if isinstance(other, Angle):
            return self._frac == other._frac
        return NotImplemented

    def __hash__(self):
        return hash(("qlx.Angle", self._frac))

    def __repr__(self):
        n, d = self.pi_fraction
        if n == 0:
            return "cudaq.logical.Angle(0)"
        magnitude = "pi" if abs(n) == 1 else f"{abs(n)}*pi"
        body = magnitude if d == 1 else f"{magnitude}/{d}"
        return f"-{body}" if n < 0 else body

    __str__ = __repr__


#: The exact angle ``pi``. Build the others by arithmetic:
#: ``cudaq.logical.pi/2`` is the Clifford point, ``cudaq.logical.pi/4`` the
#: magic point, and ``-cudaq.logical.pi/4`` its inverse.
pi = Angle(1, 1)
