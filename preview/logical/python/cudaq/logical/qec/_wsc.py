# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Private builders for high-rate WSC deformations and fixed artifacts.

The measurement construction is the Webster--Smith--Cohen (WSC) deformation
from arXiv:2511.15989, generalized to a Hermitian Pauli string by local
Clifford conjugation.  The emitted checks are the construction: modified base
stabilizers, one chi check per requested-support carrier, kappa interface
qubits, and gamma kernel checks.  They are not detached certificate metadata.

Added kappa edges form a star over the chi vertices.  This gives a simple
analytic boundary-Cheeger lower bound h>=1 and connects otherwise separate
logical factors, at the honest cost of one potentially high-weight chi check.
The code-deformation algebra is exact; circuit distance remains explicitly
uncertified because the current realization uses a serial bare-ancilla check
extractor rather than claiming a verified hook-safe schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, Signature
from typing import Any

from ..codes.pinnacle import pinnacle_gb, pinnacle_gb_instance
from cudaq.logical.programs.decorators import objective
from cudaq.logical.types.values import logical_qubit
from cudaq.logical.ops._impl import (
    cx,
    extract_syndrome,
    h,
    mpp,
    mz,
    parity,
    prepare as logical_prepare,
    reset,
    s,
    sdg,
    tick,
    x,
    z,
)
from ..gadgets import prepare_plus, prepare_zero
from cudaq.logical.codes import (
    Block,
    CarrierRoleMap,
    Code,
    Distance,
    Encoding,
    PatchTransform,
)
from cudaq.logical.algebra.clifford import CliffordAction
from cudaq.logical.gadgets import (
    GadgetDefinition,
    gadget,
    patch,
)
from cudaq.logical.algebra.pauli import (
    X,
    Y,
    Z,
)
from cudaq.logical.types.semantic import (
    plus,
    zero,
)
from ..std import idle as idle_objective


def _rref(rows: tuple[int, ...]) -> tuple[int, ...]:
    basis: dict[int, int] = {}
    for value in rows:
        row = value
        while row:
            pivot = row.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = row
                break
            row ^= basis[pivot]
    for pivot in sorted(basis):
        for other in tuple(basis):
            if other != pivot and (basis[other] >> pivot) & 1:
                basis[other] ^= basis[pivot]
    return tuple(basis[pivot] for pivot in sorted(basis, reverse=True))


def _rank(rows: tuple[int, ...]) -> int:
    return len(_rref(rows))


def _in_span(row: int, rows: tuple[int, ...]) -> bool:
    return _rank(rows + (row,)) == _rank(rows)


def _in_reduced_span(row: int, basis: tuple[int, ...]) -> bool:
    for pivot_row in basis:
        pivot = pivot_row.bit_length() - 1
        if (row >> pivot) & 1:
            row ^= pivot_row
    return row == 0


def _permute_bits(row: int, permutation: tuple[int, ...]) -> int:
    output = 0
    while row:
        bit = row & -row
        qubit = bit.bit_length() - 1
        output |= 1 << permutation[qubit]
        row ^= bit
    return output


def _gb_candidate_permutations(ell: int) -> tuple[tuple[int, ...], ...]:
    """Return Adam's complete declared circulant/fold candidate family."""

    candidates: list[tuple[int, ...]] = [tuple(range(2 * ell))]
    for shift in range(ell):
        candidates.append(
            tuple(ell +
                  (-index + shift) % ell if qubit < ell else (-index + shift) %
                  ell for qubit in range(2 * ell) for index in (qubit % ell,)))
        candidates.append(
            tuple(sector * ell + (-index + shift) % ell
                  for qubit in range(2 * ell)
                  for sector, index in (divmod(qubit, ell),)))
        candidates.append(
            tuple(ell +
                  (index + shift) % ell if qubit < ell else (index - shift) %
                  ell for qubit in range(2 * ell) for index in (qubit % ell,)))
    return tuple(dict.fromkeys(candidates))


def _nullspace(rows: tuple[int, ...], columns: int) -> tuple[int, ...]:
    reduced = _rref(rows)
    pivots = {row.bit_length() - 1: row for row in reduced}
    output = []
    for free in (column for column in range(columns) if column not in pivots):
        vector = 1 << free
        for pivot, row in pivots.items():
            if (row >> free) & 1:
                vector |= 1 << pivot
        output.append(vector)
    return tuple(output)


def _symplectic(left: "PauliRow", right: "PauliRow") -> int:
    return ((left.x & right.z).bit_count() + (left.z & right.x).bit_count()) & 1


@dataclass(frozen=True, slots=True)
class PauliRow:
    """``i**phase X**x Z**z`` with an optional diagnostic label."""

    x: int
    z: int
    phase: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if self.x < 0 or self.z < 0:
            raise ValueError("Pauli masks must be nonnegative")
        object.__setattr__(self, "phase", self.phase % 4)

    @property
    def support(self) -> tuple[int, ...]:
        mask = self.x | self.z
        return tuple(
            index for index in range(mask.bit_length()) if (mask >> index) & 1)

    @property
    def hermitian_sign(self) -> int:
        delta = (self.phase - (self.x & self.z).bit_count()) % 4
        if delta not in (0, 2):
            raise ValueError(f"Pauli row {self.label!r} is not Hermitian")
        return 1 if delta == 0 else -1

    def packed(self, width: int) -> int:
        return self.x | (self.z << width)

    def with_label(self, label: str) -> "PauliRow":
        return PauliRow(self.x, self.z, self.phase, label)

    def shifted(self, offset: int) -> "PauliRow":
        return PauliRow(self.x << offset, self.z << offset, self.phase,
                        self.label)

    def __mul__(self, other: "PauliRow") -> "PauliRow":
        return PauliRow(
            self.x ^ other.x,
            self.z ^ other.z,
            self.phase + other.phase + 2 * ((self.z & other.x).bit_count() & 1),
            self.label or other.label,
        )


def _fold_image(row: int, permutation: tuple[int, ...]) -> PauliRow:
    """Conjugate one physical X row by fixed-site S and paired CZ layers."""

    image = PauliRow(0, 0)
    while row:
        bit = row & -row
        qubit = bit.bit_length() - 1
        row ^= bit
        image = image * PauliRow(
            1 << qubit,
            1 << permutation[qubit],
            int(permutation[qubit] == qubit),
        )
    return image


@dataclass(frozen=True, slots=True)
class CliffordSupport:
    """Precise support status for one searched global Clifford realization."""

    supported: bool
    operation: str
    reason: str
    candidates_checked: int
    stabilizer_preserving_candidates: int
    canonical_candidates: int
    candidate_family: str
    objective_action: CliffordAction


@dataclass(frozen=True, slots=True)
class JointMeasurementEvidence:
    """Independent evidence facets for a WSC logical measurement."""

    checks_commute: bool
    chi_product_equals_requested: bool
    merged_rank: int
    expected_merged_rank: int
    removes_exactly_one_logical: bool
    requested_in_stabilizer_span: bool
    proper_factor_in_stabilizer_span: tuple[bool, ...]
    boundary_cheeger_lower_bound: int
    expansion_method: str
    code_distance: Distance
    circuit_distance: Distance

    @property
    def algebraically_valid(self) -> bool:
        return (self.checks_commute and self.chi_product_equals_requested and
                self.removes_exactly_one_logical and
                self.requested_in_stabilizer_span and
                not any(self.proper_factor_in_stabilizer_span))


@dataclass(frozen=True, slots=True)
class WSCMeasurementPlan:
    """One exact WSC merged-code plan and its live-patch Mark III gadget."""

    code: Any
    terms: tuple[tuple[int, str], ...]
    requested: PauliRow
    base_checks: tuple[PauliRow, ...]
    merged_checks: tuple[PauliRow, ...]
    chi_checks: tuple[PauliRow, ...]
    gamma_checks: tuple[PauliRow, ...]
    selected_base_checks: tuple[int, ...]
    incidence: tuple[int, ...]
    kappa_count: int
    star_kappa_count: int
    rounds: int
    evidence: JointMeasurementEvidence
    gadget: GadgetDefinition


def _base_stabilizers(code) -> tuple[PauliRow, ...]:
    output = []
    for index, row in enumerate(code.stabilizer_basis.rows):
        x = sum(int(bit) << qubit for qubit, bit in enumerate(row[:code.n]))
        z = sum(int(bit) << qubit for qubit, bit in enumerate(row[code.n:]))
        output.append(PauliRow(x, z, 0, f"base_{index}"))
    return tuple(output)


def _audit_global_clifford(code, ell: int, kind: str) -> CliffordSupport:
    """Audit every declared GB fold against one exact canonical action.

    Preserving the stabilizer group, or merely inducing some symplectic
    logical action, is insufficient.  The candidate must induce H or S on
    every declared logical port with the canonical signs.
    """

    if kind not in ("h", "s"):
        raise ValueError("global Clifford audit supports h or s")
    hx = tuple(sum(1 << qubit for qubit in row) for row in code.hx)
    hz = tuple(sum(1 << qubit for qubit in row) for row in code.hz)
    lx = tuple(sum(1 << qubit for qubit in row) for row in code.lx)
    lz = tuple(sum(1 << qubit for qubit in row) for row in code.lz)
    hx_basis = _rref(hx)
    hz_basis = _rref(hz)
    candidates = _gb_candidate_permutations(ell)
    stabilizer_preserving = 0
    canonical = 0

    for permutation in candidates:
        if kind == "h":
            preserved = all(
                _in_reduced_span(_permute_bits(row, permutation), hz_basis)
                for row in hx) and all(
                    _in_reduced_span(_permute_bits(row, permutation), hx_basis)
                    for row in hz)
            if not preserved:
                continue
            stabilizer_preserving += 1
            exact = True
            for logical in range(code.k):
                image_z = _permute_bits(lx[logical], permutation)
                image_x = _permute_bits(lz[logical], permutation)
                z_coordinates = sum(
                    ((image_z & lx[index]).bit_count() & 1) << index
                    for index in range(code.k))
                x_coordinates = sum(
                    ((image_x & lz[index]).bit_count() & 1) << index
                    for index in range(code.k))
                if z_coordinates != 1 << logical or x_coordinates != 1 << logical:
                    exact = False
                    break
            canonical += int(exact)
            continue

        if any(permutation[permutation[index]] != index
               for index in range(code.n)):
            continue
        check_images = tuple(_fold_image(row, permutation) for row in hx)
        preserved = all(
            _in_reduced_span(image.x, hx_basis) and
            _in_reduced_span(image.z, hz_basis) and image.phase == 0
            for image in check_images)
        if not preserved:
            continue
        stabilizer_preserving += 1
        exact = True
        for logical in range(code.k):
            image = _fold_image(lx[logical], permutation)
            target = PauliRow(lx[logical], lz[logical], 1)
            difference = image * target
            if not (_in_reduced_span(difference.x, hx_basis) and
                    _in_reduced_span(difference.z, hz_basis) and
                    difference.phase == 0):
                exact = False
                break
        canonical += int(exact)

    operation = kind.upper()
    return CliffordSupport(
        supported=canonical > 0,
        operation=kind,
        reason=
        (f"{canonical} exact canonical {operation}^tensor(k) candidate(s) found"
         if canonical else
         (f"no exact canonical {operation}^tensor(k) action among the "
          "complete declared GB circulant/fold candidate family")),
        candidates_checked=len(candidates),
        stabilizer_preserving_candidates=stabilizer_preserving,
        canonical_candidates=canonical,
        candidate_family="gb_circulant_fold",
        objective_action=_canonical_global_action(kind, code.k),
    )


def _canonical_global_action(kind: str, logicals: int) -> CliffordAction:
    if kind == "h":
        images = tuple((0, 1 << index, 0) for index in range(logicals)) + tuple(
            (1 << index, 0, 0) for index in range(logicals))
    elif kind == "s":
        images = tuple(
            (1 << index, 1 << index, 0) for index in range(logicals)) + tuple(
                (0, 1 << index, 0) for index in range(logicals))
    else:
        raise ValueError("canonical global action supports h or s")
    return CliffordAction.from_images(
        images,
        ports=tuple(range(logicals)),
        evidence=("canonical-global",),
    )


def _logical_row(code, logical: int, pauli: str) -> PauliRow:
    lx = sum(1 << qubit for qubit in code.lx[logical])
    lz = sum(1 << qubit for qubit in code.lz[logical])
    if pauli == "X":
        return PauliRow(lx, 0, 0, f"X{logical}")
    if pauli == "Z":
        return PauliRow(0, lz, 0, f"Z{logical}")
    if pauli == "Y":
        return PauliRow(lx, lz, 1, f"Y{logical}")
    raise ValueError("logical Pauli must be X, Y, or Z")


def _requested_row(code,
                   terms,
                   *,
                   sign: int = 1) -> tuple[PauliRow, tuple[PauliRow, ...]]:
    if sign not in (-1, 1):
        raise ValueError("Pauli-product sign must be +1 or -1")
    requested = PauliRow(0, 0, 0 if sign > 0 else 2, "requested")
    factors = []
    for logical, pauli in terms:
        factor = _logical_row(code, logical, pauli)
        factors.append(factor)
        requested = requested * factor
    return requested.with_label("requested"), tuple(factors)


def _build_wsc(code, terms, rounds, *, sign: int = 1):
    requested, factors = _requested_row(code, terms, sign=sign)
    requested.hermitian_sign
    support = requested.support
    if not support:
        raise ValueError("logical product reduced to identity")
    base = _base_stabilizers(code)

    selected = []
    incidence_rows = [0] * len(support)
    for check_index, check in enumerate(base):
        local = []
        for row_index, qubit in enumerate(support):
            anticommutes = ((((requested.x >> qubit) & 1) and
                             ((check.z >> qubit) & 1)) ^
                            (((requested.z >> qubit) & 1) and
                             ((check.x >> qubit) & 1)))
            if anticommutes:
                local.append(row_index)
        if local:
            column = len(selected)
            selected.append(check_index)
            for row_index in local:
                incidence_rows[row_index] |= 1 << column

    # A star is an explicit O(w) expander: every cut S with |S|<=w/2 has at
    # least |S| star edges crossing it.  Its edges also join disconnected
    # logical factors, preventing any factor's chi subset from cancelling all
    # kappa support on its own.
    star_start = len(selected)
    for row_index in range(1, len(support)):
        column = star_start + row_index - 1
        incidence_rows[0] |= 1 << column
        incidence_rows[row_index] |= 1 << column
    star_count = max(0, len(support) - 1)
    kappa_count = len(selected) + star_count
    width = code.n + kappa_count

    selected_columns = {check: column for column, check in enumerate(selected)}
    modified_base = []
    for check_index, check in enumerate(base):
        column = selected_columns.get(check_index)
        z = check.z if column is None else check.z | (1 << (code.n + column))
        modified_base.append(PauliRow(check.x, z, check.phase, check.label))

    requested_sign = requested.hermitian_sign
    chi = []
    for row_index, (qubit,
                    incidence_row) in enumerate(zip(support, incidence_rows)):
        local_x = ((requested.x >> qubit) & 1) << qubit
        local_z = ((requested.z >> qubit) & 1) << qubit
        local_phase = 1 if local_x and local_z else 0
        if row_index == 0 and requested_sign < 0:
            local_phase += 2
        chi.append(
            PauliRow(
                local_x | (incidence_row << code.n),
                local_z,
                local_phase,
                f"chi_{row_index}",
            ))
    gamma = tuple(
        PauliRow(0, vector << code.n, 0, f"gamma_{index}")
        for index, vector in enumerate(
            _nullspace(tuple(incidence_rows), kappa_count)))
    merged = tuple(modified_base) + tuple(chi) + gamma

    commute = all(
        _symplectic(left, right) == 0
        for index, left in enumerate(merged)
        for right in merged[index + 1:])
    chi_product = PauliRow(0, 0)
    for row in chi:
        chi_product = chi_product * row
    chi_equals = (chi_product.x == requested.x and
                  chi_product.z == requested.z and
                  (chi_product.phase - requested.phase) % 4 == 0)
    packed = tuple(row.packed(width) for row in merged)
    merged_rank = _rank(packed)
    expected_rank = width - (code.k - 1)
    requested_extended = requested.packed(width)
    proper = tuple(
        _in_span(factor.packed(width), packed)
        for factor in (factors if len(factors) > 1 else ()))
    evidence = JointMeasurementEvidence(
        checks_commute=commute,
        chi_product_equals_requested=chi_equals,
        merged_rank=merged_rank,
        expected_merged_rank=expected_rank,
        removes_exactly_one_logical=merged_rank == expected_rank,
        requested_in_stabilizer_span=_in_span(requested_extended, packed),
        proper_factor_in_stabilizer_span=proper,
        boundary_cheeger_lower_bound=1,
        expansion_method="explicit star kappa edges; analytic cut bound",
        code_distance=Distance.unknown(
            "WSC expansion is certified, but logical irreducibility and the "
            "joint-bridge distance hypotheses are not machine-certified"),
        circuit_distance=Distance.unknown(
            "serial bare-ancilla stabilizer extraction has no hook-safe fault proof"
        ),
    )
    if not evidence.algebraically_valid:
        raise ValueError(f"WSC merged-code certificate failed: {evidence!r}")
    return (
        requested,
        base,
        merged,
        tuple(chi),
        gamma,
        tuple(selected),
        tuple(incidence_rows),
        kappa_count,
        star_count,
        evidence,
    )


def _physical_measurement(state, row: PauliRow, check_ancilla: int, *,
                          record: str):
    x_only = tuple(qubit for qubit in row.support
                   if (row.x >> qubit) & 1 and not (row.z >> qubit) & 1)
    y_sites = tuple(qubit for qubit in row.support
                    if (row.x >> qubit) & 1 and (row.z >> qubit) & 1)
    if x_only:
        state = h(state.frame[x_only])
    if y_sites:
        state = sdg(state.frame[y_sites])
        state = h(state.frame[y_sites])
    state = reset(state.frame[(check_ancilla,)])
    if row.hermitian_sign < 0:
        state = x(state.frame[(check_ancilla,)])
    for qubit in row.support:
        state = cx(state.frame[(qubit,)], state.frame[(check_ancilla,)])
    if x_only:
        state = h(state.frame[x_only])
    if y_sites:
        state = h(state.frame[y_sites])
        state = s(state.frame[y_sites])
    return mz(state.frame[(check_ancilla,)], record=record)


def _measurement_objective(code, terms, name, *, sign: int = 1):
    parameter_names = tuple(f"q{logical}" for logical, _ in terms)

    def intent(*values):
        product = None
        for value, (_, pauli) in zip(values, terms):
            factor = {"X": X, "Y": Y, "Z": Z}[pauli](value)
            product = factor if product is None else product @ factor
        return mpp(product if sign > 0 else -product)

    intent.__name__ = intent.__qualname__ = f"{name}_objective"
    intent.__signature__ = Signature(
        tuple(
            Parameter(
                parameter,
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=logical_qubit,
            ) for parameter in parameter_names),
        return_annotation=tuple[tuple([logical_qubit] * len(terms) + [bool])],
    )
    intent.__annotations__ = {
        **{
            parameter: logical_qubit for parameter in parameter_names
        },
        "return": tuple[tuple([logical_qubit] * len(terms) + [bool])],
    }
    return objective(intent, name=f"{name}_objective")


def _global_preparation_objective(code, logical_state, flip, name: str):

    def intent():
        values = []
        for _ in range(code.k):
            value = logical_prepare(state=logical_state)
            values.append(flip(value))
        return tuple(values)

    result_annotation = tuple[tuple([logical_qubit] * code.k)]
    intent.__name__ = intent.__qualname__ = f"{name}_objective"
    intent.__signature__ = Signature((), return_annotation=result_annotation)
    intent.__annotations__ = {"return": result_annotation}
    return objective(intent, name=intent.__name__)


def _global_cnot_objective(code, name: str):
    controls = tuple(f"control{index}" for index in range(code.k))
    targets = tuple(f"target{index}" for index in range(code.k))
    names = controls + targets

    def intent(*values):
        values = list(values)
        for logical in range(code.k):
            values[logical], values[code.k + logical] = cx(
                values[logical], values[code.k + logical])
        return tuple(values)

    result_annotation = tuple[tuple([logical_qubit] * (2 * code.k))]
    intent.__name__ = intent.__qualname__ = f"{name}_objective"
    intent.__signature__ = Signature(
        tuple(
            Parameter(
                item, Parameter.POSITIONAL_OR_KEYWORD, annotation=logical_qubit)
            for item in names),
        return_annotation=result_annotation,
    )
    intent.__annotations__ = {
        **{
            item: logical_qubit for item in names
        }, "return": result_annotation
    }
    return objective(intent, name=intent.__name__)


def _measurement_gadget(
    code,
    encoding,
    terms,
    *,
    requested,
    base_checks,
    merged_checks,
    chi_checks,
    kappa_count,
    rounds,
    evidence,
    name,
    sign=1,
):
    check_ancilla = code.n + kappa_count
    frame_size = check_ancilla + 1
    data_support = tuple(range(code.n))
    scratch = tuple(range(code.n, frame_size))
    roles = CarrierRoleMap(active=data_support, scratch=scratch)
    transform = PatchTransform(
        name=f"{name}_frame",
        source=encoding,
        destination=encoding,
        frame=Block(data=frame_size),
        source_support=data_support,
        destination_support=data_support,
        source_roles=roles,
        destination_roles=roles,
        logical_map=tuple(range(code.k)),
        evidence=
        "arXiv:2511.15989v1 WSC deformation; exact rank/span certificate",
    )
    intent = _measurement_objective(code, terms, name, sign=sign)

    def realization(state):
        state = reset(state.frame[scratch])
        for check_index, row in enumerate(base_checks):
            state, _ = _physical_measurement(state,
                                             row,
                                             check_ancilla,
                                             record=f"pre_base_{check_index}")
        final = None
        chi_offset = len(base_checks)
        for round_index in range(rounds):
            current = []
            for check_index, row in enumerate(merged_checks):
                state, bit = _physical_measurement(
                    state,
                    row,
                    check_ancilla,
                    record=f"merge_{round_index}_{check_index}",
                )
                current.append(bit)
            final = tuple(current[chi_offset:chi_offset + len(chi_checks)])
            tick()
        if final is None:
            raise AssertionError(
                "WSC measurement requires at least one merge round")
        outcome = parity(*final)
        for check_index, row in enumerate(base_checks):
            state, _ = _physical_measurement(state,
                                             row,
                                             check_ancilla,
                                             record=f"split_base_{check_index}")
        if kappa_count:
            state, _ = mz(
                state.frame[tuple(range(code.n, code.n + kappa_count))],
                record="split_kappa_z",
            )
        return state, outcome

    realization.__name__ = realization.__qualname__ = name
    realization.__annotations__ = {
        "state": patch[encoding],
        "return": tuple[patch[encoding], bool],
    }
    logical_ports = {
        getattr(intent.operands, f"q{logical}"):
            getattr(encoding.ports, encoding.logical_ports[logical])
        for logical, _ in terms
    }
    return gadget(
        realization,
        implements=intent,
        logical_ports=logical_ports,
        transform=transform,
        name=name,
        metadata={
            "construction":
                "Webster-Smith-Cohen merged-code measurement",
            "requested_pauli":
                "".join(pauli for _, pauli in terms),
            "requested_phase":
                requested.phase,
            "merge_rounds":
                rounds,
            "kappa_qubits":
                kappa_count,
            "merged_checks":
                len(merged_checks),
            "chi_checks":
                len(chi_checks),
            "checks_commute":
                evidence.checks_commute,
            "rank_exact":
                evidence.removes_exactly_one_logical,
            "proper_factors_hidden":
                not any(evidence.proper_factor_in_stabilizer_span),
            "boundary_cheeger_lower_bound":
                evidence.boundary_cheeger_lower_bound,
            "code_distance_status":
                evidence.code_distance.status,
            "circuit_distance_status":
                evidence.circuit_distance.status,
            "extractor":
                "serial_bare_ancilla_unverified_fault_distance",
        },
    )


class _CSSDefinitionBuilder:
    """Private constructor for one fixed high-rate CSS definition set."""

    def __init__(self, preset: str | int):
        self.instance = pinnacle_gb_instance(preset)
        self.code = pinnacle_gb(self.instance.name)
        self.encoding = self.code.default_encoding

    def prepare_zero(self):
        return prepare_zero(self.encoding,
                            name=f"{self.code.name}_prepare_zero")

    def prepare_plus(self):
        return prepare_plus(self.encoding,
                            name=f"{self.code.name}_prepare_plus")

    def _flipped_preparation(self, *, plus_basis: bool):
        base = self.prepare_plus() if plus_basis else self.prepare_zero()
        encoding = self.encoding
        code = self.code
        flip_rows = code.lz if plus_basis else code.lx
        flip_mask = 0
        for row in flip_rows:
            flip_mask ^= sum(1 << qubit for qubit in row)
        flip_support = tuple(
            qubit for qubit in range(code.n) if (flip_mask >> qubit) & 1)
        suffix = "minus" if plus_basis else "one"
        operation = z if plus_basis else x
        intent = _global_preparation_objective(
            code,
            plus if plus_basis else zero,
            operation,
            f"{code.name}_prepare_{suffix}",
        )

        def realization(block):
            block = base(block)
            return operation(block.data[flip_support])

        realization.__name__ = realization.__qualname__ = intent.name
        realization.__annotations__ = {
            "block": patch[encoding],
            "return": patch[encoding],
        }
        return gadget(
            realization,
            implements=intent,
            name=realization.__name__,
            metadata={
                "logical_eigenvalue": -1,
                "logical_flips": code.k
            },
        )

    def prepare_one(self):
        return self._flipped_preparation(plus_basis=False)

    def prepare_minus(self):
        return self._flipped_preparation(plus_basis=True)

    def memory(self, rounds: int | None = None):
        rounds = self.instance.logical_cycle_rounds if rounds is None else rounds
        if not isinstance(rounds, int) or isinstance(rounds,
                                                     bool) or rounds <= 0:
            raise ValueError("rounds must be a positive int")
        encoding = self.encoding
        code = self.code

        def realization(block):
            for round_index in range(rounds):
                block, _ = extract_syndrome(
                    block, record=f"{code.name}_memory_{round_index}")
            return block

        realization.__name__ = realization.__qualname__ = (
            f"{code.name}_memory_r{rounds}")
        realization.__annotations__ = {
            "block": patch[encoding],
            "return": patch[encoding],
        }
        return gadget(
            realization,
            implements=idle_objective,
            name=realization.__name__,
            metadata={"syndrome_rounds": rounds},
        )

    def h_evidence(self) -> CliffordSupport:
        return _audit_global_clifford(self.code, self.instance.ell, "h")

    def s_evidence(self) -> CliffordSupport:
        return _audit_global_clifford(self.code, self.instance.ell, "s")

    def transversal_cx(self) -> GadgetDefinition:
        encoding = self.encoding
        code = self.code
        intent = _global_cnot_objective(code, f"{code.name}_transversal_cx")

        def realization(control, target):
            return cx(control.data, target.data)

        realization.__name__ = realization.__qualname__ = f"{code.name}_transversal_cx"
        realization.__annotations__ = {
            "control": patch[encoding],
            "target": patch[encoding],
            "return": tuple[patch[encoding], patch[encoding]],
        }
        return gadget(
            realization,
            implements=intent,
            logical_ports={
                **{
                    f"control{index}":
                        f"control.{encoding.logical_ports[index]}" for index in range(code.k)
                },
                **{
                    f"target{index}": f"target.{encoding.logical_ports[index]}" for index in range(code.k)
                },
            },
            name=realization.__name__,
            metadata={
                "construction": "blockwise transversal CX",
                "physical_pairs": code.n,
                "logical_pairs": code.k,
            },
        )

    def build_wsc_measurement(
        self,
        terms: tuple[tuple[int, str], ...],
        *,
        rounds: int | None = None,
        sign: int = 1,
    ) -> WSCMeasurementPlan:
        """Build and emit one within-block arbitrary logical Pauli measurement."""

        if not isinstance(terms, tuple) or not terms:
            raise ValueError(
                "terms must be a nonempty tuple of (logical, Pauli)")
        if sign not in (-1, 1):
            raise ValueError("Pauli-product sign must be +1 or -1")
        normalized = []
        occupied = set()
        for term in terms:
            if not isinstance(term, tuple) or len(term) != 2:
                raise TypeError("each term must be (logical, Pauli)")
            logical, pauli = term
            if not isinstance(logical, int) or isinstance(logical, bool):
                raise TypeError("logical index must be an int")
            if not 0 <= logical < self.code.k:
                raise ValueError("logical index out of range")
            if pauli not in ("X", "Y", "Z"):
                raise ValueError("logical Pauli must be X, Y, or Z")
            if logical in occupied:
                raise ValueError("a logical port may occur only once")
            occupied.add(logical)
            normalized.append((logical, pauli))
        normalized_terms = tuple(normalized)
        rounds = self.instance.logical_cycle_rounds if rounds is None else rounds
        if not isinstance(rounds, int) or isinstance(rounds,
                                                     bool) or rounds <= 0:
            raise ValueError("rounds must be a positive int")
        (
            requested,
            base,
            merged,
            chi,
            gamma,
            selected,
            incidence,
            kappa_count,
            star_count,
            evidence,
        ) = _build_wsc(self.code, normalized_terms, rounds, sign=sign)
        name = (f"{self.code.name}_wsc_" + "_".join(
            f"{pauli.lower()}{logical}" for logical, pauli in normalized_terms))
        definition = _measurement_gadget(
            self.code,
            self.encoding,
            normalized_terms,
            requested=requested,
            base_checks=base,
            merged_checks=merged,
            chi_checks=chi,
            kappa_count=kappa_count,
            rounds=rounds,
            evidence=evidence,
            name=name,
            sign=sign,
        )
        return WSCMeasurementPlan(
            code=self.code,
            terms=normalized_terms,
            requested=requested,
            base_checks=base,
            merged_checks=merged,
            chi_checks=chi,
            gamma_checks=gamma,
            selected_base_checks=selected,
            incidence=incidence,
            kappa_count=kappa_count,
            star_kappa_count=star_count,
            rounds=rounds,
            evidence=evidence,
            gadget=definition,
        )


@dataclass(frozen=True, slots=True)
class _GenericCSSInstance:
    logical_cycle_rounds: int


class _HighRateCSSBuilder(_CSSDefinitionBuilder):
    """Private Mark III definition builder for an arbitrary native CSS code.

    General CSS codes get preparation, repeated memory, transversal CX, and
    exact within-block WSC product measurement. Global H/S are exposed only
    when the code is a Pinnacle GB instance whose declared fold family can be
    exhaustively audited; otherwise the result is precisely unsupported.
    """

    def __init__(self, value, *, logical_cycle_rounds: int | None = None):
        if isinstance(value, Encoding):
            encoding = value
            code = value.code
        elif isinstance(value, Code):
            code = value
            encoding = value.default_encoding
        else:
            raise TypeError(
                "high-rate CSS definitions require a CSS Code or Encoding")
        if not all(hasattr(code, field) for field in ("hx", "hz", "lx", "lz")):
            raise TypeError("high-rate CSS definitions require CSS code data")
        if logical_cycle_rounds is None:
            logical_cycle_rounds = code.d.value
        if (not isinstance(logical_cycle_rounds, int) or
                isinstance(logical_cycle_rounds, bool) or
                logical_cycle_rounds <= 0):
            raise ValueError(
                "logical_cycle_rounds must be provided as a positive int when "
                "the code has no scalar distance value")
        self.code = code
        self.encoding = encoding
        self.instance = _GenericCSSInstance(logical_cycle_rounds)

    def _global_support(self, kind: str) -> CliffordSupport:
        ell = self.code.metadata.get("ell")
        if (self.code.metadata.get("family") == "pinnacle_generalized_bicycle"
                and isinstance(ell, int) and not isinstance(ell, bool)):
            return _audit_global_clifford(self.code, ell, kind)
        return CliffordSupport(
            supported=False,
            operation=kind,
            reason=("no physical global Clifford candidate family was supplied "
                    "for this CSS encoding"),
            candidates_checked=0,
            stabilizer_preserving_candidates=0,
            canonical_candidates=0,
            candidate_family="none_declared",
            objective_action=_canonical_global_action(kind, self.code.k),
        )

    def h_evidence(self) -> CliffordSupport:
        return self._global_support("h")

    def s_evidence(self) -> CliffordSupport:
        return self._global_support("s")


__all__ = []
