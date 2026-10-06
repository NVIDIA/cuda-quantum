# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Private exact synthesis support for architecture-owned RUS protocols.

This module implements the constructive core of the one-ancilla axial-rotation
scheme of Bocharov, Roetteler, and Svore (arXiv:1404.5320).  It deliberately
returns an immutable description; the Pinnacle architecture owns conversion of
that description into QLX protocols, scratch allocation, WSC readout, and
retry.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RotationColumn:
    """One chronological exact Pauli rotation in the lifted RUS circuit.

    ``quarter_turns`` measures the QLX rotation angle in units of ``pi/4``.
    Odd columns are the actual non-Clifford gates and consume one T state;
    even columns are exact Clifford work and deliberately do not.
    """

    data: bool
    ancilla: str | None
    quarter_turns: int
    consumes_t_state: bool = False

    def __post_init__(self) -> None:
        if not self.data and self.ancilla is None:
            raise ValueError("a RUS rotation column cannot be identity")
        if self.ancilla not in (None, "X", "Y", "Z"):
            raise ValueError("RUS ancilla Pauli must be X, Y, Z, or None")
        if (not isinstance(self.quarter_turns, int) or
                isinstance(self.quarter_turns, bool) or
                self.quarter_turns == 0 or abs(self.quarter_turns) > 4):
            raise ValueError("RUS rotation must use one to four quarter turns")
        if self.consumes_t_state and abs(self.quarter_turns) != 1:
            raise ValueError(
                "one T state realizes exactly one signed quarter turn")
        if not self.consumes_t_state and self.quarter_turns % 2:
            raise ValueError("an odd RUS rotation must consume one T state")


@dataclass(frozen=True, slots=True)
class RUSDesign:
    """Certified cyclotomic design and concrete quarter-turn attempt."""

    requested_angle: float
    effective_angle: float
    requested_precision: float
    achieved_error: float
    success_probability: float
    denominator_exponent: int
    z_coefficients: tuple[int, int, int, int]
    y_coefficients: tuple[int, int, int, int]
    normalization: tuple[int, int]
    single_qubit_t_count: int
    lifted_t_count: int
    columns: tuple[RotationColumn, ...]
    sha256: str

    @property
    def t_states_per_attempt(self) -> int:
        return sum(column.consumes_t_state for column in self.columns)

    @property
    def clifford_rotations_per_attempt(self) -> int:
        return len(self.columns) - self.t_states_per_attempt


def _dependencies():
    try:
        import mpmath
        from pygridsynth.diophantine import diophantine_dyadic
        from pygridsynth.domega_unitary import DOmegaUnitary
        from pygridsynth.normal_form import Clifford, NormalForm, Syllable
        from pygridsynth.quantum_circuit import QuantumCircuit
        from pygridsynth.quantum_gate import HGate, SGate, SXGate, TGate, WGate
        from pygridsynth.ring import DOmega, DRootTwo, ZOmega, ZRootTwo
        from pygridsynth.synthesis_of_cliffordT import decompose_domega_unitary
    except ImportError as error:
        raise ImportError(
            "Pinnacle arbitrary-angle RUS synthesis requires PyGridSynth; "
            "install `cudaq-logical[synthesis]`") from error
    return (
        mpmath,
        diophantine_dyadic,
        DOmegaUnitary,
        Clifford,
        HGate,
        SGate,
        SXGate,
        TGate,
        WGate,
        DOmega,
        DRootTwo,
        ZOmega,
        ZRootTwo,
        decompose_domega_unitary,
        NormalForm,
        Syllable,
        QuantumCircuit,
    )


def _finite_positive(value: float, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _canonical_angle(angle: float) -> float:
    if not isinstance(angle, (int, float)) or isinstance(angle, bool):
        raise TypeError("RUS angle must be a finite number")
    result = float(angle)
    if not math.isfinite(result):
        raise ValueError("RUS angle must be a finite number")
    return (result + math.pi) % (2.0 * math.pi) - math.pi


def _z_coefficients(z) -> tuple[int, int, int, int]:
    return (int(z.a), int(z.b), int(z.c), int(z.d))


def _approximate_phase(angle: float, precision: float, *, work_digits: int):
    dependencies = _dependencies()
    mp = dependencies[0]
    ZOmega = dependencies[11]
    with mp.workdps(work_digits):
        theta = mp.mpf(str(angle))
        target = mp.exp(1j * theta)
        vector = mp.matrix([
            mp.cos(theta / 2) - mp.sin(theta / 2),
            mp.sqrt(2) * mp.cos(theta / 2),
            mp.cos(theta / 2) + mp.sin(theta / 2),
            mp.sqrt(2) * mp.sin(theta / 2),
        ])
        for refinement in range(10):
            tolerance = mp.mpf(str(precision)) / (2**refinement)
            for max_coefficient in (100, 1_000, 10_000, 100_000, 1_000_000):
                relation = mp.pslq(
                    vector,
                    tol=tolerance,
                    maxcoeff=max_coefficient,
                    maxsteps=10_000,
                )
                if relation is None:
                    continue
                coefficients = tuple(int(value) for value in relation)
                divisor = math.gcd(*coefficients)
                if divisor:
                    coefficients = tuple(
                        value // divisor for value in coefficients)
                first = next((value for value in coefficients if value), 0)
                if first < 0:
                    coefficients = tuple(-value for value in coefficients)
                z = ZOmega(*coefficients)
                value = z.to_complex
                if abs(value) == 0:
                    continue
                achieved = abs(value.conjugate() / value - target)
                if achieved <= precision:
                    return z, float(achieved)
        raise RuntimeError(
            "bounded PSLQ search did not find a cyclotomic RUS phase within "
            f"precision {precision:.6e}")


def _normalization_candidates(radius: int):
    if radius == 0:
        yield (1, 0)
        return
    for a in range(-radius, radius + 1):
        for b in range(-radius, radius + 1):
            if max(abs(a), abs(b)) != radius or (a == 0 and b == 0):
                continue
            if a < 0 or (a == 0 and b < 0):
                continue
            yield (a, b)


def _normalize(z, *, work_digits: int):
    dependencies = _dependencies()
    mp = dependencies[0]
    diophantine_dyadic = dependencies[1]
    DOmegaUnitary = dependencies[2]
    TGate = dependencies[7]
    DOmega = dependencies[9]
    DRootTwo = dependencies[10]
    ZRootTwo = dependencies[12]
    decompose_domega_unitary = dependencies[13]
    best = None
    with mp.workdps(work_digits):
        for radius in range(13):
            for a, b in _normalization_candidates(radius):
                multiplier = ZRootTwo(a, b)
                rz = z * multiplier
                squared = ZRootTwo.from_zomega(rz * rz.conj)
                norm = mp.mpf(squared.a) + mp.mpf(squared.b) * mp.sqrt(2)
                if norm <= 0:
                    continue
                exponent = int(mp.ceil(mp.log(norm, 2)))
                probability = norm / (2**exponent)
                if probability <= mp.mpf("0.5") or probability > 1:
                    continue
                residue = DRootTwo.from_int(2**exponent) - DRootTwo.from_zomega(
                    rz * rz.conj)
                solution = diophantine_dyadic(residue, seed=0)
                if not isinstance(solution, DOmega):
                    continue
                y = DOmega(solution.u, exponent)
                unitary = DOmegaUnitary(
                    DOmega(rz, exponent),
                    -y.conj,
                    0,
                )
                try:
                    circuit = decompose_domega_unitary(
                        unitary,
                        [1],
                        up_to_phase=False,
                    )
                except (AssertionError, ValueError):
                    # PyGridSynth rejects a non-reduced common-denominator
                    # presentation.  It is not a usable exact V candidate.
                    continue
                t_count = sum(isinstance(gate, TGate) for gate in circuit)
                score = float(t_count / probability)
                candidate = (
                    score,
                    t_count,
                    -float(probability),
                    radius,
                    a,
                    b,
                    unitary,
                    y,
                    exponent,
                    float(probability),
                )
                if best is None or candidate[:6] < best[:6]:
                    best = candidate
            if best is not None and radius >= 4:
                break
    if best is None:
        raise RuntimeError(
            "bounded deterministic normalization did not find a solvable RUS "
            "norm equation with one-round success probability above 1/2")
    _, t_count, _, _, a, b, unitary, y, exponent, probability = best
    return unitary, y, (a, b), exponent, probability, t_count


@lru_cache(maxsize=1)
def _cliffords():
    _, _, _, Clifford, *_ = _dependencies()
    return tuple(
        Clifford(a, b, c, d)
        for a in range(3)
        for b in range(2)
        for c in range(4)
        for d in range(8))


def _domega_key(unitary) -> tuple[str, str, int]:
    return repr(unitary.z), repr(unitary.w), int(unitary.n)


def _signed_t_circuit(sign: int):
    """Return exact matrix-order gates for T or T-dagger."""

    dependencies = _dependencies()
    SGate = dependencies[5]
    TGate = dependencies[7]
    QuantumCircuit = dependencies[16]
    circuit = QuantumCircuit()
    if sign < 0:
        # S^3 T = T^7 = T-dagger exactly, including cyclotomic phase.
        for _ in range(3):
            circuit.append(SGate(1))
    circuit.append(TGate(1))
    return circuit


@lru_cache(maxsize=1)
def _clifford_by_unitary():
    dependencies = _dependencies()
    DOmegaUnitary = dependencies[2]
    return {
        _domega_key(DOmegaUnitary.from_circuit(clifford.to_circuit([1]))):
            clifford for clifford in _cliffords()
    }


def _t_code_initial_states(factor):
    """All exact ``factor T = G T^sign C`` finite decompositions."""

    dependencies = _dependencies()
    DOmegaUnitary = dependencies[2]
    QuantumCircuit = dependencies[16]
    clifford_by_unitary = _clifford_by_unitary()
    output = []
    for left in _cliffords():
        for sign in (1, -1):
            circuit = QuantumCircuit()
            circuit += _signed_t_circuit(-sign)
            circuit += left.inv().to_circuit([1])
            circuit += factor.to_circuit([1])
            circuit += _signed_t_circuit(1)
            residual = clifford_by_unitary.get(
                _domega_key(DOmegaUnitary.from_circuit(circuit)))
            if residual is not None:
                output.append((left, (sign,), residual))
    return tuple(output)


@lru_cache(maxsize=1)
def _t_code_transitions():
    """Finite exact rewrites ``D T = H T^sign C`` for Cliffords D, C."""

    dependencies = _dependencies()
    DOmegaUnitary = dependencies[2]
    HGate = dependencies[4]
    TGate = dependencies[7]
    QuantumCircuit = dependencies[16]

    right_by_unitary: dict[tuple[str, str, int], list[tuple[int, Any]]] = {}
    for sign in (1, -1):
        for clifford in _cliffords():
            circuit = QuantumCircuit()
            circuit.append(HGate(1))
            circuit += _signed_t_circuit(sign)
            circuit += clifford.to_circuit([1])
            right_by_unitary.setdefault(
                _domega_key(DOmegaUnitary.from_circuit(circuit)), []).append(
                    (sign, clifford))

    output = {}
    for clifford in _cliffords():
        circuit = clifford.to_circuit([1])
        circuit.append(TGate(1))
        output[(
            clifford.a,
            clifford.b,
            clifford.c,
            clifford.d,
        )] = tuple(
            sorted(
                right_by_unitary.get(
                    _domega_key(DOmegaUnitary.from_circuit(circuit)), ()),
                key=lambda item: (
                    -item[0],
                    item[1].a,
                    item[1].b,
                    item[1].c,
                    item[1].d,
                ),
            ))
    return output


def _left_clifford_inverse(unitary, clifford):
    dependencies = _dependencies()
    HGate = dependencies[4]
    SGate = dependencies[5]
    SXGate = dependencies[6]
    WGate = dependencies[8]
    result = unitary
    for gate in clifford.to_circuit([1]):
        if isinstance(gate, HGate):
            result = (
                result.renew_denomexp(result.k +
                                      1).mul_by_H_from_left().reduce_denomexp())
        elif isinstance(gate, SGate):
            result = result.mul_by_S_power_from_left(-1).reduce_denomexp()
        elif isinstance(gate, SXGate):
            result = result.mul_by_X_from_left().reduce_denomexp()
        elif isinstance(gate, WGate):
            result = result.mul_by_W_power_from_left(-1).reduce_denomexp()
        else:  # pragma: no cover - Clifford.to_circuit has a closed gate set.
            raise RuntimeError(
                f"unsupported Clifford gate {type(gate).__name__}")
    return result


def _as_clifford(unitary):
    if unitary.k != 0:
        return None
    return _clifford_by_unitary().get(_domega_key(unitary))


def _strip_t_code(unitary, remaining: int, failed: set[tuple[Any, ...]]):
    key = (repr(unitary.z), repr(unitary.w), int(unitary.n), remaining)
    if key in failed:
        return None
    if remaining == 0:
        clifford = _as_clifford(unitary)
        if clifford is not None:
            return (), clifford
        failed.add(key)
        return None
    choices = []
    for sign in (1, -1):
        successor = (unitary.mul_by_T_power_from_left(-sign).renew_denomexp(
            unitary.k + 1).mul_by_H_from_left().reduce_denomexp())
        if successor.k <= unitary.k:
            choices.append((successor.k, -sign, successor))
    for _, negated_sign, successor in sorted(choices):
        sign = -negated_sign
        suffix = _strip_t_code(successor, remaining - 1, failed)
        if suffix is not None:
            signs, clifford = suffix
            return (sign, *signs), clifford
    failed.add(key)
    return None


def _searched_t_code_normal_form(unitary, t_count: int):
    """Fast denominator-reduction path used by the common exact forms."""

    # A failed (exact unitary, remaining depth) subproblem is independent of
    # the outer left-Clifford seed.  Sharing the negative memo across all 24
    # seeds avoids rediscovering convergent dead branches while preserving the
    # first successful candidate and its deterministic order.
    failed = set()
    for left in _cliffords():
        reduced = _left_clifford_inverse(unitary, left)
        result = _strip_t_code(reduced, t_count, failed)
        if result is not None:
            signs, right = result
            return left, tuple(signs), right
    return None


def _t_code_normal_forms(unitary, t_count: int):
    """Lazily enumerate exact decorated T-codes admitted by the BRS lift."""

    # The denominator-reduction search is both cheaper and deliberately first
    # in the candidate order.  Yield it before constructing the exhaustive
    # finite automaton: the common path certifies this candidate and never
    # needs the remaining normal forms.  If certification rejects it, iteration
    # resumes below and preserves the previous exhaustive fallback exactly.
    searched = _searched_t_code_normal_form(unitary, t_count)
    if searched is not None:
        yield searched

    dependencies = _dependencies()
    Clifford = dependencies[3]
    decompose = dependencies[13]
    NormalForm = dependencies[14]
    Syllable = dependencies[15]
    normal = NormalForm.from_circuit(decompose(unitary, [1], up_to_phase=False))
    if len(normal.syllables) != t_count or not normal.syllables:
        raise RuntimeError(
            "exact RUS unitary has an inconsistent T normal form")

    identity = Clifford(0, 0, 0, 0)
    hadamard = Clifford.from_str("H")
    phase = Clifford.from_str("S")
    factors = {
        Syllable.T: identity,
        Syllable.HT: hadamard,
        Syllable.SHT: phase * hadamard,
    }
    states = list(_t_code_initial_states(factors[normal.syllables[0]]))
    transitions = _t_code_transitions()
    for syllable in normal.syllables[1:]:
        successors = []
        for left, signs, residual in states:
            combined = residual * factors[syllable]
            key = (combined.a, combined.b, combined.c, combined.d)
            successors.extend((left, (*signs, sign), right)
                              for sign, right in transitions.get(key, ()))
        by_residual = {}
        for left, signs, residual in successors:
            key = (
                left.a,
                left.b,
                left.c,
                left.d,
                residual.a,
                residual.b,
                residual.c,
                residual.d,
            )
            previous = by_residual.get(key)
            if previous is None or signs < previous[1]:
                by_residual[key] = (left, signs, residual)
        states = list(by_residual.values())
        if not states:
            raise RuntimeError(
                "failed to derive the exact decorated T-code normal form")
    candidates = sorted(
        (
            # The finite automaton represents a word ending in its last T.
            # A BRS T-code ends every syllable in H, so insert H into the
            # right Clifford: T C = T H (H C).
            (left, signs, hadamard * residual * normal.c)
            for left, signs, residual in states),
        key=lambda item: (
            item[0].a,
            item[0].b,
            item[0].c,
            item[0].d,
            item[1],
            item[2].a,
            item[2].b,
            item[2].c,
            item[2].d,
        ),
    )
    unique = {}
    for left, signs, right in candidates:
        key = (
            left.a,
            left.b,
            left.c,
            left.d,
            signs,
            right.a,
            right.b,
            right.c,
            right.d,
        )
        unique.setdefault(key, (left, signs, right))
    for candidate in unique.values():
        if candidate != searched:
            yield candidate


def _local_clifford_algebraic(clifford) -> tuple[str, ...]:
    """Primitive algebraic word for one ancilla-local Clifford.

    PyGridSynth returns its circuit in matrix-product order.  Global omega
    phases are irrelevant for an uncontrolled local gate and are omitted.
    """

    _, _, _, _, HGate, SGate, SXGate, _, WGate, *_ = _dependencies()
    names = {
        HGate: "ancilla_h",
        SGate: "ancilla_s",
        SXGate: "ancilla_x",
    }
    output = []
    for gate in clifford.to_circuit([1]):
        if isinstance(gate, WGate):
            continue
        try:
            output.append(names[type(gate)])
        except KeyError as error:  # pragma: no cover - closed library gate set.
            raise RuntimeError(
                f"unsupported local Clifford gate {type(gate).__name__}"
            ) from error
    return tuple(output)


def _matrix_key(matrix: np.ndarray) -> tuple[tuple[float, float], ...]:
    return tuple((round(float(value.real), 10), round(float(value.imag), 10))
                 for value in matrix.reshape(-1))


def _phase_power_algebraic(exponent: int) -> tuple[str, ...]:
    """Minimal algebraic ``T_data**exponent`` word, modulo global phase."""

    exponent %= 8
    candidates = []
    for t_power in (-1, 0, 1):
        if (exponent - t_power) % 2:
            continue
        s_power = ((exponent - t_power) // 2) % 4
        if s_power > 2:
            s_power -= 4
        candidates.append((abs(t_power), abs(s_power), t_power, s_power))
    _, _, t_power, s_power = min(candidates)
    output = [
        "data_s" if s_power > 0 else "data_sdg" for _ in range(abs(s_power))
    ]
    if t_power:
        output.append("data_t" if t_power > 0 else "data_tdg")
    return tuple(output)


_CSM_ALGEBRAIC = (
    "cx",
    "ancilla_tdg",
    "cx",
    "ancilla_t",
)
_CH_ALGEBRAIC = (
    "ancilla_sdg",
    "ancilla_h",
    "ancilla_tdg",
    "cx",
    "ancilla_t",
    "ancilla_h",
    "ancilla_s",
)


def _controlled_pauli_algebraic(
    phase: int,
    pauli: str,
) -> tuple[str, ...]:
    """Exact controlled ``i**phase * pauli`` as Clifford primitives."""

    if pauli == "i":
        controlled = ()
    elif pauli == "x":
        controlled = ("cx",)
    elif pauli == "z":
        controlled = ("cz",)
    elif pauli == "y":
        # CY = S_control CX CZ in algebraic matrix-product order.
        controlled = ("data_s", "cx", "cz")
    else:  # pragma: no cover - generated only by the finite factor table.
        raise RuntimeError(f"unsupported controlled Pauli {pauli!r}")
    return (*_phase_power_algebraic(2 * phase), *controlled)


def _controlled_representative_algebraic(
    omega_phase: int,
    representative: str,
) -> tuple[str, ...]:
    """Controlled representative from the finite Clifford double classes."""

    if representative == "i":
        return _phase_power_algebraic(omega_phase)
    if representative == "s":
        body = _CSM_ALGEBRAIC
        phase = omega_phase + 1
    elif representative == "h":
        body = _CH_ALGEBRAIC
        phase = omega_phase
    elif representative == "sh":
        body = (*_CSM_ALGEBRAIC, *_CH_ALGEBRAIC)
        phase = omega_phase + 1
    elif representative == "hs":
        body = (*_CH_ALGEBRAIC, *_CSM_ALGEBRAIC)
        phase = omega_phase + 1
    elif representative == "hsh":
        body = ("ancilla_h", *_CSM_ALGEBRAIC, "ancilla_h")
        phase = omega_phase + 1
    else:  # pragma: no cover - generated only by the finite factor table.
        raise RuntimeError(
            f"unsupported controlled-Clifford representative {representative!r}"
        )
    return (*_phase_power_algebraic(phase), *body)


@lru_cache(maxsize=1)
def _controlled_clifford_factors():
    """Find a parsimonious exact controlled circuit for every Clifford.

    The finite search realizes the constructive observation in BRS Appendix B:
    a controlled one-qubit Clifford is Clifford-equivalent to one of a small
    set of controlled representatives.  Pauli wrappers are themselves
    Clifford, while the representatives cost at most five T gates.
    """

    _, _, DOmegaUnitary, *_ = _dependencies()
    identity = np.eye(2, dtype=complex)
    x = np.array(((0, 1), (1, 0)), dtype=complex)
    y = np.array(((0, -1j), (1j, 0)), dtype=complex)
    z = np.diag((1, -1)).astype(complex)
    h = np.array(((1, 1), (1, -1)), dtype=complex) / math.sqrt(2.0)
    s = np.diag((1, 1j)).astype(complex)
    omega = np.exp(1j * math.pi / 4.0)
    paulis = tuple(
        (phase, name, (1j**phase) * matrix)
        for phase in range(4)
        for name, matrix in (("i", identity), ("x", x), ("y", y), ("z", z)))
    representatives = (
        ("i", identity),
        ("s", s),
        ("h", h),
        ("sh", s @ h),
        ("hs", h @ s),
        ("hsh", h @ s @ h),
    )

    candidates: dict[
        tuple[tuple[float, float], ...],
        list[tuple[int, int, tuple[int, str, int, str, int, str]]],
    ] = {}
    for left_phase, left_name, left in paulis:
        for omega_phase in range(8):
            for representative, middle in representatives:
                middle = (omega**omega_phase) * middle
                for right_phase, right_name, right in paulis:
                    factor = (
                        left_phase,
                        left_name,
                        omega_phase,
                        representative,
                        right_phase,
                        right_name,
                    )
                    algebraic = (
                        *_controlled_pauli_algebraic(left_phase, left_name),
                        *_controlled_representative_algebraic(
                            omega_phase, representative),
                        *_controlled_pauli_algebraic(right_phase, right_name),
                    )
                    t_count = sum(token in {
                        "data_t",
                        "data_tdg",
                        "ancilla_t",
                        "ancilla_tdg",
                    } for token in algebraic)
                    candidates.setdefault(_matrix_key(left @ middle @ right),
                                          []).append(
                                              (t_count, len(algebraic), factor))

    output = {}
    for clifford in _cliffords():
        matrix = np.asarray(
            DOmegaUnitary.from_circuit(clifford.to_circuit(
                [1])).to_complex_matrix.tolist(),
            dtype=complex,
        )
        matches = candidates.get(_matrix_key(matrix), ())
        if not matches:
            raise RuntimeError(
                f"no exact controlled-Clifford factor for {clifford!r}")
        selected = min(matches)
        if selected[0] > 5:
            raise RuntimeError(
                "controlled-Clifford factor exceeded five T gates")
        output[(clifford.a, clifford.b, clifford.c, clifford.d)] = selected[2]
    return output


def _controlled_clifford_algebraic(clifford) -> tuple[str, ...]:
    factor = _controlled_clifford_factors()[(clifford.a, clifford.b, clifford.c,
                                             clifford.d)]
    left_phase, left, omega_phase, middle, right_phase, right = factor
    return (
        *_controlled_pauli_algebraic(left_phase, left),
        *_controlled_representative_algebraic(omega_phase, middle),
        *_controlled_pauli_algebraic(right_phase, right),
    )


def _h_columns() -> tuple[RotationColumn, ...]:
    # H = Rz(pi/2) Rx(pi/2) Rz(pi/2), up to global phase.
    return (
        RotationColumn(False, "Z", 2),
        RotationColumn(False, "X", 2),
        RotationColumn(False, "Z", 2),
    )


def _cz_columns() -> tuple[RotationColumn, ...]:
    # CZ = Rz_d(pi/2) Rz_a(pi/2) Rzz(-pi/2), up to global phase.
    return (
        RotationColumn(True, None, 2),
        RotationColumn(False, "Z", 2),
        RotationColumn(True, "Z", -2),
    )


def _cx_columns() -> tuple[RotationColumn, ...]:
    return (*_h_columns(), *_cz_columns(), *_h_columns())


def _chronological_primitive(token: str) -> tuple[RotationColumn, ...]:
    if token == "ancilla_t":
        return (RotationColumn(False, "Z", 1, True),)
    if token == "ancilla_tdg":
        return (RotationColumn(False, "Z", -1, True),)
    if token == "data_t":
        return (RotationColumn(True, None, 1, True),)
    if token == "data_tdg":
        return (RotationColumn(True, None, -1, True),)
    if token == "ancilla_s":
        return (RotationColumn(False, "Z", 2),)
    if token == "ancilla_sdg":
        return (RotationColumn(False, "Z", -2),)
    if token == "data_s":
        return (RotationColumn(True, None, 2),)
    if token == "data_sdg":
        return (RotationColumn(True, None, -2),)
    if token == "ancilla_h":
        return _h_columns()
    if token == "ancilla_x":
        return (RotationColumn(False, "X", 4),)
    if token == "cx":
        return _cx_columns()
    if token == "cz":
        return _cz_columns()
    raise RuntimeError(f"unsupported lifted RUS primitive {token!r}")


def _simplify_columns(
    columns: tuple[RotationColumn, ...],) -> tuple[RotationColumn, ...]:
    """Combine adjacent same-axis powers without charging Clifford work as T."""

    output: list[RotationColumn] = []
    index = 0
    while index < len(columns):
        column = columns[index]
        end = index + 1
        total = column.quarter_turns
        while (end < len(columns) and columns[end].data == column.data and
               columns[end].ancilla == column.ancilla):
            total += columns[end].quarter_turns
            end += 1
        residue = total % 8
        if residue > 4:
            residue -= 8
        if residue:
            if residue % 2 == 0:
                output.append(
                    RotationColumn(column.data, column.ancilla, residue))
            else:
                resource_turn = 1 if residue > 0 else -1
                clifford_turns = residue - resource_turn
                if clifford_turns:
                    output.append(
                        RotationColumn(
                            column.data,
                            column.ancilla,
                            clifford_turns,
                        ))
                output.append(
                    RotationColumn(
                        column.data,
                        column.ancilla,
                        resource_turn,
                        True,
                    ))
        index = end
    return tuple(output)


def _lift(unitary, t_count: int):
    _, _, _, Clifford, *_ = _dependencies()
    for g1, signs, g2 in _t_code_normal_forms(unitary, t_count):
        inverse_signs = tuple(-value for value in reversed(signs))
        hadamard = Clifford.from_str("H")
        g3 = g2.inv() * hadamard
        g4 = hadamard * g1.inv()
        phase = sum(
            (right - left) // 2 for left, right in zip(signs, inverse_signs))
        decorated_g4 = Clifford(g4.a, g4.b, g4.c, g4.d + phase)
        g5 = g3 * g1.inv()
        g6 = g2.inv() * decorated_g4

        algebraic: list[str] = [*_controlled_clifford_algebraic(g5)]
        algebraic.extend(_local_clifford_algebraic(g1))
        for left, right in zip(signs, inverse_signs):
            if left != right:
                algebraic.append("cx")
            algebraic.append("ancilla_t" if left > 0 else "ancilla_tdg")
            if left != right:
                algebraic.append("cx")
            algebraic.append("ancilla_h")
        algebraic.extend(_local_clifford_algebraic(g2))
        algebraic.extend(_controlled_clifford_algebraic(g6))

        columns = _simplify_columns(
            tuple(column for token in reversed(algebraic)
                  for column in _chronological_primitive(token)))
        lifted_t_count = sum(column.consumes_t_state for column in columns)
        if lifted_t_count > t_count + 9:
            continue
        if _lift_matches(unitary, columns):
            return columns, lifted_t_count
    raise RuntimeError(
        "no exact decorated T-code produced a certified BRS J(V) lift")


def _column_matrix(column: RotationColumn) -> np.ndarray:
    identity = np.eye(2, dtype=complex)
    x = np.array(((0, 1), (1, 0)), dtype=complex)
    y = np.array(((0, -1j), (1j, 0)), dtype=complex)
    z = np.diag((1, -1)).astype(complex)
    data = z if column.data else identity
    ancilla = {None: identity, "X": x, "Y": y, "Z": z}[column.ancilla]
    pauli = np.kron(data, ancilla)
    half_angle = column.quarter_turns * math.pi / 8.0
    return (math.cos(half_angle) * np.eye(4, dtype=complex) -
            1j * math.sin(half_angle) * pauli)


def _projective_error(left: np.ndarray, right: np.ndarray) -> float:
    overlap = np.vdot(right.reshape(-1), left.reshape(-1))
    if abs(overlap) == 0:
        return math.inf
    phase = overlap / abs(overlap)
    return float(np.linalg.norm(left - phase * right, ord=2))


def _lift_matches(unitary, columns) -> bool:
    realized = np.eye(4, dtype=complex)
    for column in columns:
        realized = _column_matrix(column) @ realized
    v = np.asarray(unitary.to_complex_matrix.tolist(), dtype=complex)
    expected = np.block([[v, np.zeros((2, 2), complex)],
                         [np.zeros((2, 2), complex),
                          v.conj().T]])
    return _projective_error(realized, expected) <= 1.0e-8


def _certify(unitary, columns, angle: float, probability: float) -> float:
    realized = np.eye(4, dtype=complex)
    for column in columns:
        realized = _column_matrix(column) @ realized
    v = np.asarray(unitary.to_complex_matrix.tolist(), dtype=complex)
    expected = np.block([[v, np.zeros((2, 2), complex)],
                         [np.zeros((2, 2), complex),
                          v.conj().T]])
    if _projective_error(realized, expected) > 1.0e-8:
        raise RuntimeError(
            "lifted RUS quarter-turn sequence failed J(V) certification")

    success = realized[np.ix_((0, 2), (0, 2))]
    failure = realized[np.ix_((1, 3), (0, 2))]
    success_probability = float(np.trace(success.conj().T @ success).real / 2.0)
    failure_probability = float(np.trace(failure.conj().T @ failure).real / 2.0)
    if not math.isclose(
            success_probability, probability, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            "lifted RUS success probability failed certification")
    if not math.isclose(
            success_probability + failure_probability, 1.0, abs_tol=1e-9):
        raise RuntimeError("lifted RUS Kraus probabilities are not normalized")
    target = np.diag((np.exp(-0.5j * angle), np.exp(0.5j * angle)))
    achieved = _projective_error(success / math.sqrt(probability), target)
    pauli_z = np.diag((1, -1)).astype(complex)
    if not 0.0 < probability < 1.0:
        raise RuntimeError(
            "lifted RUS probability must lie strictly inside (0, 1)")
    if _projective_error(failure / math.sqrt(1.0 - probability),
                         pauli_z) > 1.0e-8:
        raise RuntimeError("lifted RUS failure branch is not Pauli-correctable")
    return achieved


@lru_cache(maxsize=128)
def synthesize(angle: float, precision: float) -> RUSDesign:
    """Synthesize and certify one bounded-input BRS axial RUS attempt."""

    effective_angle = _canonical_angle(angle)
    precision = _finite_positive(precision, name="RUS precision")
    work_digits = max(60, math.ceil(-math.log10(precision)) + 30)
    z, phase_error = _approximate_phase(
        effective_angle,
        precision,
        work_digits=work_digits,
    )
    (
        unitary,
        y,
        normalization,
        exponent,
        probability,
        t_count,
    ) = _normalize(z, work_digits=work_digits)
    columns, lifted_t_count = _lift(unitary, t_count)
    achieved_error = _certify(
        unitary,
        columns,
        effective_angle,
        probability,
    )
    if achieved_error > precision or phase_error > precision:
        raise RuntimeError(
            "certified RUS attempt missed its requested projective error bound")
    rz = unitary.z.u
    payload = "|".join((
        f"{effective_angle:.17g}",
        f"{precision:.17g}",
        str(exponent),
        ",".join(map(str, _z_coefficients(rz))),
        ",".join(map(str, _z_coefficients(y.u))),
        ";".join(f"{int(column.data)}:{column.ancilla or 'I'}:"
                 f"{column.quarter_turns}:{int(column.consumes_t_state)}"
                 for column in columns),
    ))
    return RUSDesign(
        requested_angle=float(angle),
        effective_angle=effective_angle,
        requested_precision=precision,
        achieved_error=achieved_error,
        success_probability=probability,
        denominator_exponent=exponent,
        z_coefficients=_z_coefficients(rz),
        y_coefficients=_z_coefficients(y.u),
        normalization=normalization,
        single_qubit_t_count=t_count,
        lifted_t_count=lifted_t_count,
        columns=columns,
        sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )


__all__: tuple[str, ...] = ()
