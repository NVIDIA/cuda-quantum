# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Python rotation synthesizers and independently checked chronological words."""

from __future__ import annotations

import hashlib
import math
import re
import sys
import threading
from dataclasses import dataclass, replace
from typing import Protocol

_ERROR_METRIC = "projective_operator_norm"


@dataclass(frozen=True, slots=True)
class CliffordTWord:
    """One normalized H/S/T word and its synthesis certificate."""

    gates: tuple[str, ...]
    requested_precision: float
    achieved_error: float
    algorithm: str
    implementation: str
    implementation_version: str
    error_metric: str = _ERROR_METRIC

    @property
    def text(self) -> str:
        return " ".join(self.gates)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def t_count(self) -> int:
        return self.gates.count("t")

    @property
    def clifford_count(self) -> int:
        return len(self.gates) - self.t_count

    def evidence(self) -> dict[str, str | int | float]:
        return {
            "synthesis_algorithm": self.algorithm,
            "synthesis_implementation": self.implementation,
            "synthesis_version": self.implementation_version,
            "synthesis_error_metric": self.error_metric,
            "synthesis_requested_precision": self.requested_precision,
            "synthesis_achieved_error": self.achieved_error,
            "synthesis_word_sha256": self.sha256,
            "synthesis_t_count": self.t_count,
            "synthesis_clifford_count": self.clifford_count,
        }


class RotationSynthesizer(Protocol):
    """Deterministic synthesis, up to global phase, into chronological words.

    ``rz`` is required. An optional ``u3(theta, phi, lam, precision)`` method
    receives the whole Euler rotation Rz(phi) Ry(theta) Rz(lam).
    """

    name: str
    version: str

    def rz(self, angle: float, precision: float) -> CliffordTWord:
        ...


_EXPANSIONS = {
    "h": ("h",),
    "s": ("s",),
    "t": ("t",),
    "sdg": ("s",) * 3,
    "tdg": ("s",) * 3 + ("t",),
    "x": ("h", "s", "s", "h"),
    "y": ("h", "s", "s", "h", "s", "s"),
    "z": ("s", "s"),
}


def _word_error(gates, kind, angles, mp=None):
    """Independently multiply the word and minimize its global-phase error."""
    import cmath

    real = float if mp is None else mp.mpf
    sqrt = math.sqrt if mp is None else mp.sqrt
    exp = cmath.exp if mp is None else mp.exp
    arg = cmath.phase if mp is None else mp.arg
    sin = math.sin if mp is None else mp.sin
    cos = math.cos if mp is None else mp.cos
    pi = math.pi if mp is None else mp.pi
    half = sqrt(real(0.5))
    matrices = {
        "h": ((half, half), (half, -half)),
        "s": ((1, 0), (0, 1j)),
        "t": ((1, 0), (0, exp(0.25j * pi))),
    }
    result = ((1, 0), (0, 1))
    for gate in gates:
        left = matrices[gate]
        result = tuple(
            tuple(
                sum(left[r][k] * result[k][c]
                    for k in range(2))
                for c in range(2))
            for r in range(2))
    values = tuple(real(value) for value in angles)
    if kind == "rz":
        target = ((exp(-0.5j * values[0]), 0), (0, exp(0.5j * values[0])))
    else:
        theta, phi, lam = values
        c, s = cos(theta / 2), sin(theta / 2)
        # Form the two Rz factors separately: adding large f64 angles can
        # overflow or erase a small angle before the transcendental call.
        p, l = exp(-0.5j * phi), exp(-0.5j * lam)
        target = ((c * p * l, -s * p * l.conjugate()),
                  (s * p.conjugate() * l, c * p.conjugate() * l.conjugate()))

    def det(matrix):
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    phase = arg(det(result) / det(target)) / 2

    def distance(candidate):
        factor = exp(1j * candidate)
        delta = tuple(
            tuple(result[r][c] - factor * target[r][c]
                  for c in range(2))
            for r in range(2))
        h00 = abs(delta[0][0])**2 + abs(delta[1][0])**2
        h11 = abs(delta[0][1])**2 + abs(delta[1][1])**2
        h01 = (delta[0][0].conjugate() * delta[0][1] +
               delta[1][0].conjugate() * delta[1][1])
        trace = h00 + h11
        determinant = max(real(0), h00 * h11 - abs(h01)**2)
        return sqrt(
            max(real(0),
                (trace + sqrt(max(real(0), trace**2 - 4 * determinant))) / 2))

    return min(distance(phase), distance(phase + pi))


def _verify_word(kind, angles, precision, word) -> CliffordTWord:
    from .gate_sets import _validate_precision

    precision = _validate_precision(precision)
    if kind not in ("rz", "u3") or len(angles) != (1 if kind == "rz" else 3):
        raise ValueError("invalid rotation synthesis request")
    if any(not math.isfinite(float(value)) for value in angles):
        raise ValueError("rotation synthesis angles must be finite")
    if not isinstance(word, CliffordTWord):
        raise TypeError("rotation synthesizer must return CliffordTWord")
    if word.error_metric != _ERROR_METRIC:
        raise ValueError(
            "rotation synthesizer must use projective_operator_norm")
    if not math.isfinite(word.achieved_error) or word.achieved_error < 0:
        raise ValueError(
            "synthesizer achieved_error must be finite and nonnegative")
    gates = []
    for gate in word.gates:
        if not isinstance(gate, str) or gate.lower() not in _EXPANSIONS:
            raise ValueError(f"unsupported Clifford+T gate {gate!r}")
        gates.extend(_EXPANSIONS[gate.lower()])
    gates = tuple(gates)
    try:
        import mpmath
    except ImportError:
        # Leave generous headroom for matrix multiplication and norm evaluation.
        # Never silently relax the requested bound to the numerical floor.
        guard = 64 * sys.float_info.epsilon * (len(gates) + 1)
        if precision < 4 * guard:
            raise ValueError(
                f"cannot verify precision {precision:g} in double precision "
                f"for a {len(gates)}-gate word; install mpmath")
        error = _word_error(gates, kind, angles)
    else:
        mp = mpmath.mp.clone()
        mp.dps = max(
            50,
            int(-math.log10(precision)) + 30 + int(math.log10(len(gates) + 1)))
        guard = mp.power(10, -mp.dps + 10) * (len(gates) + 1)
        error = _word_error(gates, kind, angles, mp)
    if error + guard > precision:
        raise ValueError(
            f"{kind} synthesis word exceeds requested precision {precision:g}: "
            f"independently checked error bound {float(error + guard):.17g}")
    return replace(word,
                   gates=gates,
                   requested_precision=precision,
                   achieved_error=float(error + guard))


# Strong references deliberately keep registered callbacks alive until shutdown.
# Object identity supports unhashable synthesizers and prevents id reuse. Human
# names are provenance, never registry keys: two differently configured objects
# with the same name/version must not replace one another's callbacks.
_registered = {}
_labels = {}
_registration_lock = threading.RLock()
_SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")


def _ensure_registered(synthesizer) -> str:
    if synthesizer is None:
        return "gridsynth"
    if isinstance(synthesizer, str):
        if not _SAFE_NAME.fullmatch(synthesizer):
            raise ValueError(
                "synthesizer registry name contains invalid characters")
        return synthesizer
    with _registration_lock:
        existing = _registered.get(id(synthesizer))
        if existing is not None:
            return existing[1]
        for field in ("name", "version"):
            value = getattr(synthesizer, field, None)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(
                    f"rotation synthesizer {field} must be a nonempty string")
        rz = getattr(synthesizer, "rz", None)
        u3 = getattr(synthesizer, "u3", None)
        if not callable(rz) or (u3 is not None and not callable(u3)):
            raise TypeError(
                "rotation synthesizer requires callable rz and optional callable u3"
            )
        name = f"python_synth_{len(_registered)}"

        def checked_rz(angle, precision):
            word = _verify_word("rz", (angle,), precision, rz(angle, precision))
            return "".join(word.gates), word.achieved_error, word.algorithm

        def checked_u3(theta, phi, lam, precision):
            word = _verify_word("u3", (theta, phi, lam), precision,
                                u3(theta, phi, lam, precision))
            return "".join(word.gates), word.achieved_error, word.algorithm

        from cudaq.mlir._mlir_libs import _qlxRuntime as runtime

        runtime.register_rotation_synthesizer(
            name, synthesizer.version, checked_rz,
            checked_u3 if u3 is not None else None)
        _registered[id(synthesizer)] = (synthesizer, name)
        _labels[name] = f"{synthesizer.name}/{synthesizer.version}"
        return name


def _synthesizer_label(name):
    return _labels.get(name, name)


class PyGridSynth:
    """Reference Python plugin using the optional pygridsynth package."""

    name = "pygridsynth"

    @property
    def version(self):
        from cudaq.logical.qec.synthesis import _pygridsynth_version
        return _pygridsynth_version()

    def rz(self, angle, precision):
        from cudaq.logical.qec.synthesis import pygridsynth_rz
        return pygridsynth_rz(angle, precision)


__all__ = ["CliffordTWord", "RotationSynthesizer", "PyGridSynth"]
