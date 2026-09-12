# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Private WSC synthesis for the curated logical-microarchitecture recipes.

It expands the Webster--Smith--Cohen merged-code algebra into ordinary typed
QLX definitions without installing process-global implementation defaults.
The architecture recipes keep exact
algebraic certificates separate from circuit-distance evidence: the serial
bare-ancilla extractor is executable, but its hook-safe fault distance is
explicitly unknown.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from inspect import Parameter, Signature
from typing import Any

import cudaq.logical.algebra as algebra
import cudaq.logical.codes as codes
import cudaq.logical.gadgets as gadgets
import cudaq.logical.ops as ops
import cudaq.logical.types as types
from cudaq.logical.programs.decorators import objective as _objective

__all__ = ()


def _metadata_tree(value):
    """Serialize typed construction evidence into public metadata values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _metadata_tree(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _metadata_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_metadata_tree(item) for item in value)
    if isinstance(value, Enum):
        return _metadata_tree(value.value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(f"WSC metadata cannot serialize {type(value).__name__}")


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
    # The rows are already bit-packed GF(2) vectors.  Eliminating them with
    # Python's native arbitrary-width XOR is exact and avoids expanding the
    # WSC matrices into millions of boxed scalar bits solely to ask for rank.
    return len(_rref(rows))


def _in_span(row: int, rows: tuple[int, ...]) -> bool:
    return _rank(rows + (row,)) == _rank(rows)


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


def _symplectic(left: "_LabeledPauli", right: "_LabeledPauli") -> int:
    return ((left.x & right.z).bit_count() + (left.z & right.x).bit_count()) & 1


@dataclass(frozen=True, slots=True)
class _LabeledPauli:
    """Thin diagnostic adapter over QLX's phase-aware Pauli element."""

    element: types.PauliGroupElement
    label: str = ""

    @classmethod
    def from_masks(
        cls,
        x: int,
        z: int,
        phase: int = 0,
        label: str = "",
        *,
        arity: int | None = None,
    ) -> "_LabeledPauli":
        width = max(1, (x | z).bit_length()) if arity is None else arity
        return cls(types.PauliGroupElement(width, x, z, phase), label)

    @property
    def x(self) -> int:
        return self.element.x_mask

    @property
    def z(self) -> int:
        return self.element.z_mask

    @property
    def phase(self) -> int:
        return self.element.phase_exponent_mod_4

    @property
    def support(self) -> tuple[int, ...]:
        mask = self.x | self.z
        indices = []
        while mask:
            bit = mask & -mask
            indices.append(bit.bit_length() - 1)
            mask ^= bit
        return tuple(indices)

    @property
    def hermitian_sign(self) -> int:
        if not self.element.is_hermitian:
            raise ValueError(f"Pauli row {self.label!r} is not Hermitian")
        delta = (self.phase - (self.x & self.z).bit_count()) % 4
        return 1 if delta == 0 else -1

    def packed(self, width: int) -> int:
        return self.x | (self.z << width)

    def with_label(self, label: str) -> "_LabeledPauli":
        return _LabeledPauli(self.element, label)

    def shifted(self, offset: int) -> "_LabeledPauli":
        return _LabeledPauli.from_masks(
            self.x << offset,
            self.z << offset,
            self.phase,
            self.label,
            arity=self.element.arity + offset,
        )

    def __mul__(self, other: "_LabeledPauli") -> "_LabeledPauli":
        return _LabeledPauli(self.element.multiply(other.element), self.label or
                             other.label)


@dataclass(frozen=True, slots=True)
class WSCMeasurementEvidence:
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
    code_distance: codes.Distance
    circuit_distance: codes.Distance

    @property
    def algebraically_valid(self) -> bool:
        return (self.checks_commute and self.chi_product_equals_requested and
                self.removes_exactly_one_logical and
                self.requested_in_stabilizer_span and
                not any(self.proper_factor_in_stabilizer_span))


@dataclass(frozen=True, slots=True)
class LogicalFrameRecovery:
    """Exact record-controlled logical recovery for one WSC split."""

    split_corrections: tuple[_LabeledPauli, ...]
    kappa_corrections: tuple[_LabeledPauli, ...]
    split_logical_coordinates: tuple[int, ...]
    kappa_logical_coordinates: tuple[int, ...]
    preserved_correlation_count: int


@dataclass(frozen=True, slots=True)
class WSCMeasurementPlan:
    """One exact WSC merged-code plan and its live-patch Mark III gadget."""

    code: codes.Code
    terms: tuple[tuple[int, str], ...]
    requested: _LabeledPauli
    base_checks: tuple[_LabeledPauli, ...]
    merged_checks: tuple[_LabeledPauli, ...]
    chi_checks: tuple[_LabeledPauli, ...]
    gamma_checks: tuple[_LabeledPauli, ...]
    selected_base_checks: tuple[int, ...]
    incidence: tuple[int, ...]
    kappa_count: int
    star_kappa_count: int
    rounds: int
    evidence: WSCMeasurementEvidence
    gadget: gadgets.GadgetDefinition


def _base_stabilizers(code) -> tuple[_LabeledPauli, ...]:
    output = []
    for index, row in enumerate(code.stabilizer_basis.rows):
        x = sum(int(bit) << qubit for qubit, bit in enumerate(row[:code.n]))
        z = sum(int(bit) << qubit for qubit, bit in enumerate(row[code.n:]))
        output.append(_LabeledPauli.from_masks(x, z, 0, f"base_{index}"))
    return tuple(output)


def _logical_row(code, logical: int, pauli: str) -> _LabeledPauli:
    lx = sum(1 << qubit for qubit in code.lx[logical])
    lz = sum(1 << qubit for qubit in code.lz[logical])
    if pauli == "X":
        return _LabeledPauli.from_masks(lx, 0, 0, f"X{logical}")
    if pauli == "Z":
        return _LabeledPauli.from_masks(0, lz, 0, f"Z{logical}")
    if pauli == "Y":
        return _LabeledPauli.from_masks(lx, lz, 1, f"Y{logical}")
    raise ValueError("logical Pauli must be X, Y, or Z")


def _requested_row(
        code,
        terms,
        *,
        sign: int = 1) -> tuple[_LabeledPauli, tuple[_LabeledPauli, ...]]:
    if sign not in (-1, 1):
        raise ValueError("Pauli-product sign must be +1 or -1")
    requested = _LabeledPauli.from_masks(0, 0, 0 if sign > 0 else 2,
                                         "requested")
    factors = []
    for logical, pauli in terms:
        factor = _logical_row(code, logical, pauli)
        factors.append(factor)
        requested = requested * factor
    return requested.with_label("requested"), tuple(factors)


def _wsc_incidence(requested, base):
    """Derive the selected checks and explicit-star incidence exactly once."""

    support = requested.support
    if not support:
        raise ValueError("logical product reduced to identity")
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
    return tuple(selected), tuple(incidence_rows), kappa_count, star_count


def _wsc_kappa_count(code, terms) -> int:
    """Return the exact scratch width without building a merged-code gadget."""

    requested, _ = _requested_row(code, terms)
    base = _base_stabilizers(code)
    _, _, kappa_count, _ = _wsc_incidence(requested, base)
    return kappa_count


def _build_wsc(code, terms, rounds, *, sign: int = 1):
    requested, factors = _requested_row(code, terms, sign=sign)
    requested.hermitian_sign
    support = requested.support
    base = _base_stabilizers(code)
    selected, incidence_rows, kappa_count, star_count = _wsc_incidence(
        requested, base)
    width = code.n + kappa_count

    selected_columns = {check: column for column, check in enumerate(selected)}
    modified_base = []
    for check_index, check in enumerate(base):
        column = selected_columns.get(check_index)
        z = check.z if column is None else check.z | (1 << (code.n + column))
        modified_base.append(
            _LabeledPauli.from_masks(check.x, z, check.phase, check.label))

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
            _LabeledPauli.from_masks(
                local_x | (incidence_row << code.n),
                local_z,
                local_phase,
                f"chi_{row_index}",
            ))
    gamma = tuple(
        _LabeledPauli.from_masks(0, vector << code.n, 0, f"gamma_{index}")
        for index, vector in enumerate(
            _nullspace(tuple(incidence_rows), kappa_count)))
    merged = tuple(modified_base) + tuple(chi) + gamma

    commute = all(
        _symplectic(left, right) == 0
        for index, left in enumerate(merged)
        for right in merged[index + 1:])
    chi_product = _LabeledPauli.from_masks(0, 0)
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
    evidence = WSCMeasurementEvidence(
        checks_commute=commute,
        chi_product_equals_requested=chi_equals,
        merged_rank=merged_rank,
        expected_merged_rank=expected_rank,
        removes_exactly_one_logical=merged_rank == expected_rank,
        requested_in_stabilizer_span=_in_span(requested_extended, packed),
        proper_factor_in_stabilizer_span=proper,
        boundary_cheeger_lower_bound=1,
        expansion_method="explicit star kappa edges; analytic cut bound",
        code_distance=codes.Distance.unknown(
            "WSC expansion is certified, but logical irreducibility and the "
            "joint-bridge distance hypotheses are not machine-certified"),
        circuit_distance=codes.Distance.unknown(
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


def _physical_logical_row(code, coordinates: int, *,
                          label: str) -> _LabeledPauli:
    x_coordinates = coordinates & ((1 << code.k) - 1)
    z_coordinates = coordinates >> code.k
    row = _LabeledPauli.from_masks(0, 0)
    for logical in range(code.k):
        if (x_coordinates >> logical) & 1:
            row = row * _logical_row(code, logical, "X")
    for logical in range(code.k):
        if (z_coordinates >> logical) & 1:
            row = row * _logical_row(code, logical, "Z")
    return _LabeledPauli.from_masks(
        row.x,
        row.z,
        row.phase + (x_coordinates & z_coordinates).bit_count(),
        label,
    )


def _rref_with_provenance(
    rows: tuple[int, ...],
    columns: int,
) -> tuple[tuple[int, int, int], ...]:
    """Reduce GF(2) rows while retaining each row's source combination."""

    work = [[row, 1 << index] for index, row in enumerate(rows)]
    rank = 0
    for column in reversed(range(columns)):
        found = next((index for index in range(rank, len(work))
                      if (work[index][0] >> column) & 1), None)
        if found is None:
            continue
        work[rank], work[found] = work[found], work[rank]
        pivot_row, pivot_sources = work[rank]
        for index in range(len(work)):
            if index != rank and (work[index][0] >> column) & 1:
                work[index][0] ^= pivot_row
                work[index][1] ^= pivot_sources
        rank += 1
    return tuple(
        (row.bit_length() - 1, row, sources) for row, sources in work if row)


def _logical_frame_recovery(
    code,
    requested: _LabeledPauli,
    incidence: tuple[int, ...],
    kappa_count: int,
    base_check_count: int,
) -> LogicalFrameRecovery:
    """Derive the terminal-kappa logical correction of a WSC split.

    Gamma checks restrict the final kappa-Z outcomes to the row space of the
    vertex/edge incidence matrix. A provenance-preserving RREF gives one data
    Pauli product for every independent kappa pivot. Its commutation with the
    canonical logical basis is the exact logical byproduct to remove.
    """

    reduced = _rref_with_provenance(incidence, kappa_count)
    expected_rank = max(0, len(incidence) - 1)
    if len(reduced) != expected_rank:
        raise ValueError(
            "WSC incidence must have connected even-column rank |support|-1")
    kappa_coordinates = [0] * kappa_count
    support = requested.support
    for pivot, _row, sources in reduced:
        byproduct = _LabeledPauli.from_masks(0, 0)
        for vertex, qubit in enumerate(support):
            if (sources >> vertex) & 1:
                bit = 1 << qubit
                byproduct = byproduct * _LabeledPauli.from_masks(
                    requested.x & bit,
                    requested.z & bit,
                    int(bool((requested.x & bit) and (requested.z & bit))),
                )
        x_coordinates = sum(
            _symplectic(byproduct, _logical_row(code, logical, "Z")) << logical
            for logical in range(code.k))
        z_coordinates = sum(
            _symplectic(_logical_row(code, logical, "X"), byproduct) << logical
            for logical in range(code.k))
        coordinates = x_coordinates | (z_coordinates << code.k)
        correction = _physical_logical_row(code,
                                           coordinates,
                                           label=f"kappa_logical_frame_{pivot}")
        if _symplectic(correction, requested):
            raise ValueError(
                "WSC logical-frame correction changes the requested outcome")
        kappa_coordinates[pivot] = coordinates

    split_coordinates = (0,) * base_check_count
    return LogicalFrameRecovery(
        split_corrections=tuple(
            _LabeledPauli.from_masks(0, 0, label=f"split_logical_frame_{index}")
            for index in range(base_check_count)),
        kappa_corrections=tuple(
            _physical_logical_row(
                code, coordinates, label=f"kappa_logical_frame_{index}")
            for index, coordinates in enumerate(kappa_coordinates)),
        split_logical_coordinates=split_coordinates,
        kappa_logical_coordinates=tuple(kappa_coordinates),
        preserved_correlation_count=2 * code.k - 1,
    )


def _controlled_frame_pauli(state, control: int, row: _LabeledPauli):
    """Apply a Pauli to frame data coherently controlled by one ancilla."""

    for qubit in row.support:
        if (row.x >> qubit) & 1:
            state = ops.cx(state.frame[(control,)], state.frame[(qubit,)])
        if (row.z >> qubit) & 1:
            state = ops.cz(state.frame[(control,)], state.frame[(qubit,)])
    return state


def _physical_measurement(
    state,
    row: _LabeledPauli,
    check_ancilla: int,
    *,
    record: str,
    recovery: _LabeledPauli | None = None,
):
    x_only = tuple(qubit for qubit in row.support
                   if (row.x >> qubit) & 1 and not (row.z >> qubit) & 1)
    y_sites = tuple(qubit for qubit in row.support
                    if (row.x >> qubit) & 1 and (row.z >> qubit) & 1)
    if x_only:
        state = ops.h(state.frame[x_only])
    if y_sites:
        state = ops.sdg(state.frame[y_sites])
        state = ops.h(state.frame[y_sites])
    state = ops.reset(state.frame[(check_ancilla,)])
    if row.hermitian_sign < 0:
        state = ops.x(state.frame[(check_ancilla,)])
    if row.support:
        # One typed pair relation represents the whole bare-ancilla parity
        # interaction.  Emitting a separate immutable patch successor for
        # every carrier made authoring and MLIR verification scale with total
        # check weight even though the relation is one commuting operation.
        state = ops.cx(
            state.frame,
            state.frame,
            pairs=tuple((qubit, check_ancilla) for qubit in row.support),
        )
    if x_only:
        state = ops.h(state.frame[x_only])
    if y_sites:
        state = ops.h(state.frame[y_sites])
        state = ops.s(state.frame[y_sites])
    if recovery is not None:
        state = _controlled_frame_pauli(state, check_ancilla, recovery)
    return ops.mz(state.frame[(check_ancilla,)], record=record)


def _row_from_symplectic(bits, width: int, *, label: str) -> _LabeledPauli:
    """Convert one QLX symplectic row into the local Pauli representation."""

    x_mask = sum(int(bit) << index for index, bit in enumerate(bits[:width]))
    z_mask = sum(int(bit) << index for index, bit in enumerate(bits[width:]))
    return _LabeledPauli.from_masks(x_mask, z_mask, 0, label)


def _measurement_objective(code, terms, name, *, sign: int = 1):
    parameter_names = tuple(f"q{logical}" for logical, _ in terms)

    def intent(*values):
        product = None
        for value, (_, pauli) in zip(values, terms):
            factor = {"X": types.X, "Y": types.Y, "Z": types.Z}[pauli](value)
            product = factor if product is None else product @ factor
        return ops.mpp(product if sign > 0 else -product)

    intent.__name__ = intent.__qualname__ = f"{name}_objective"
    intent.__signature__ = Signature(
        tuple(
            Parameter(
                parameter,
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=types.logical_qubit,
            ) for parameter in parameter_names),
        return_annotation=tuple[tuple([types.logical_qubit] * len(terms) +
                                      [bool])],
    )
    intent.__annotations__ = {
        **{
            parameter: types.logical_qubit for parameter in parameter_names
        },
        "return": tuple[tuple([types.logical_qubit] * len(terms) + [bool])],
    }
    return _objective(intent, name=f"{name}_objective")


def _measurement_gadget(
    code,
    encoding,
    terms,
    *,
    requested,
    base_checks,
    merged_checks,
    chi_checks,
    incidence,
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
    roles = codes.CarrierRoleMap(active=data_support, scratch=scratch)
    transform = codes.PatchTransform(
        name=f"{name}_frame",
        source=encoding,
        destination=encoding,
        frame=codes.Block(data=frame_size),
        source_support=data_support,
        destination_support=data_support,
        source_roles=roles,
        destination_roles=roles,
        logical_map=tuple(range(code.k)),
        evidence=
        "arXiv:2511.15989v1 WSC deformation; exact rank/span certificate",
    )
    intent = _measurement_objective(code, terms, name, sign=sign)
    frame_recovery = _logical_frame_recovery(
        code,
        requested,
        incidence,
        kappa_count,
        len(base_checks),
    )

    def realization(state):
        state = ops.reset(state.frame[scratch])
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
            ops.tick()
        if final is None:
            raise AssertionError(
                "WSC measurement requires at least one merge round")
        outcome = ops.parity(*final)
        for check_index, (row, anti_stabilizer) in enumerate(
                zip(base_checks, code.anti_stabilizers.rows)):
            state, _ = _physical_measurement(
                state,
                row,
                check_ancilla,
                record=f"split_base_{check_index}",
                recovery=_row_from_symplectic(
                    anti_stabilizer,
                    code.n,
                    label=f"split_recovery_{check_index}",
                ) * frame_recovery.split_corrections[check_index],
            )
        if kappa_count:
            for index, correction in enumerate(
                    frame_recovery.kappa_corrections):
                state = _controlled_frame_pauli(state, code.n + index,
                                                correction)
            state, _ = ops.mz(
                state.frame[tuple(range(code.n, code.n + kappa_count))],
                record="split_kappa_z",
            )
        return state, outcome

    realization.__name__ = realization.__qualname__ = name
    realization.__annotations__ = {
        "state": types.patch[encoding],
        "return": tuple[types.patch[encoding], bool],
    }
    logical_ports = {
        getattr(intent.operands, f"q{logical}"): encoding.ports[logical]
        for logical, _ in terms
    }
    return gadgets.gadget(
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
            "split_recovery":
                "coherent canonical anti-stabilizer plus kappa-derived logical frame",
            "logical_frame_recovery":
                _metadata_tree(frame_recovery),
        },
    )


@dataclass(frozen=True, slots=True)
class WSCMeasurementBundle:
    """One inspectable WSC realization, analysis profile, and evidence bundle."""

    product: types.PauliProduct
    realization: gadgets.GadgetDefinition
    analysis: gadgets.GadgetProfile
    evidence: WSCMeasurementEvidence
    rounds: int
    data_code: codes.Code
    auxiliary_code: codes.Code | None = None
    diagnostics: Any = field(default=None, repr=False, compare=False)

    @property
    def kappa_qubits(self) -> int:
        return self.diagnostics.kappa_count

    @property
    def merged_check_count(self) -> int:
        return len(self.diagnostics.merged_checks)

    @property
    def chi_check_count(self) -> int:
        return len(self.diagnostics.chi_checks)


def measurement_profile(diagnostics, *, name: str) -> gadgets.GadgetProfile:
    """Derive the complete output-syndrome boundary for a WSC gadget."""

    gadget = diagnostics.gadget
    boundary = {output.syndrome: False for output in gadget.outputs.blocks}
    evidence = diagnostics.evidence
    return gadgets.GadgetProfile(
        gadget,
        boundary=boundary,
        boundary_complete=True,
        name=name,
        metadata={
            "evidence_schema":
                "qlx.architectures.wsc-algebra/v1",
            "checks_commute":
                evidence.checks_commute,
            "chi_product_equals_requested":
                evidence.chi_product_equals_requested,
            "merged_rank":
                evidence.merged_rank,
            "expected_merged_rank":
                evidence.expected_merged_rank,
            "removes_exactly_one_logical":
                evidence.removes_exactly_one_logical,
            "requested_in_stabilizer_span":
                evidence.requested_in_stabilizer_span,
            "proper_factor_in_stabilizer_span":
                evidence.proper_factor_in_stabilizer_span,
            "boundary_cheeger_lower_bound":
                evidence.boundary_cheeger_lower_bound,
            "expansion_method":
                evidence.expansion_method,
            "code_distance_status":
                evidence.code_distance.status,
            "code_distance_reason":
                evidence.code_distance.reason,
            "circuit_distance_status":
                evidence.circuit_distance.status,
            "circuit_distance_reason":
                evidence.circuit_distance.reason,
        },
    )


def measurement_result_parity(diagnostics) -> gadgets.ProfileParity:
    """Return the final WSC application-result parity."""

    chi_offset = len(diagnostics.base_checks)
    final_round = diagnostics.rounds - 1
    records = tuple(
        diagnostics.gadget.record(
            f"merge_{final_round}_{chi_offset + index}.data0")
        for index in range(len(diagnostics.chi_checks)))
    parity = records[0]
    for record in records[1:]:
        parity ^= record
    return gadgets.ProfileParity.from_value(parity)
