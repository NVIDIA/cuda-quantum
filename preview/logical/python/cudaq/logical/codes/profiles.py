# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from inspect import signature
import json
import math
import re
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from ..errors import InvalidCodeAlgebra
from cudaq.logical._core.immutable import ImmutableValue
from cudaq.logical.algebra.clifford import CliffordAction
from cudaq.logical.algebra.gf2 import (
    GF2Matrix,
    _normalize_binary_value,
    _normalize_binary_values,
    _row_bits,
)
from cudaq.logical.architecture.logical import (
    LogicalValueGroup,
    LogicalValueRef,
)

from .selection import _canonical_label_index, _deep_freeze
from .structure import Block, CarrierRoleMap, PatchTransform
from .distance import (
    Distance,
    _BoundaryMaps,
    _code_symplectic_rows,
    _coordinates_in_basis_many,
    _declared_pauli_row,
    _derive_boundary_maps,
    _distance,
    _independent_row_indices,
    _independent_rows,
    _in_span,
    _row_relations,
    _solve_gf2,
    _support_bits,
    _support_rows,
    _symplectic_product,
    _symplectic_product_bits,
    _validate_symplectic_closure,
    _xor_rows,
)


@dataclass(frozen=True, slots=True)
class GaugeMeasurementMap:
    operators: GF2Matrix
    stabilizer_map: GF2Matrix

    def __post_init__(self) -> None:
        if not isinstance(self.operators, GF2Matrix):
            raise TypeError("gauge operators must be a GF2Matrix")
        if not isinstance(self.stabilizer_map, GF2Matrix):
            raise TypeError("gauge-to-stabilizer map must be a GF2Matrix")
        if self.stabilizer_map.ncols != self.operators.nrows:
            raise ValueError(
                "gauge-to-stabilizer width must equal measured gauge count")
        if self.stabilizer_map.rank != self.stabilizer_map.nrows:
            raise ValueError("gauge-to-stabilizer rows must be independent")

    @property
    def recovered_stabilizers(self) -> GF2Matrix:
        return self.stabilizer_map @ self.operators


@dataclass(frozen=True, slots=True)
class MeasurementPhase:
    name: str
    measured_gauges: GF2Matrix
    instantaneous_stabilizers: GF2Matrix
    temporal_recovery: GF2Matrix | None = None
    input_epoch: str | None = None
    output_epoch: str | None = None
    logical_map: Mapping[str, str] | None = None
    logical_action: CliffordAction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("measurement phase name must be nonempty")
        for label, matrix in (
            ("measured_gauges", self.measured_gauges),
            ("instantaneous_stabilizers", self.instantaneous_stabilizers),
        ):
            if not isinstance(matrix, GF2Matrix):
                raise TypeError(
                    f"measurement phase {label} must be a GF2Matrix")
        if self.temporal_recovery is not None and not isinstance(
                self.temporal_recovery, GF2Matrix):
            raise TypeError("phase temporal_recovery must be a GF2Matrix")
        if self.logical_map is not None and not isinstance(
                self.logical_map, Mapping):
            raise TypeError("measurement phase logical_map must be a mapping")
        if self.logical_action is not None and not isinstance(
                self.logical_action, CliffordAction):
            raise TypeError(
                "measurement phase logical_action must be a CliffordAction")
        logical_map = dict(self.logical_map or {})
        if any(not isinstance(key, str) or not key or
               not isinstance(value, str) or not value
               for key, value in logical_map.items()):
            raise TypeError(
                "measurement phase logical_map must map nonempty strings to "
                "nonempty strings")
        object.__setattr__(self, "logical_map", MappingProxyType(logical_map))


@dataclass(frozen=True, slots=True)
class RecordLogicalMap:
    """Record-coordinate updates for one paired protected periodic basis.

    Columns use the canonical phase-major gauge-record order. Rows use
    ``names`` order on both the X and Z sides, matching the period closure's
    canonical ``(X..., Z...)`` order. A zero row means that basis coordinate
    has no record-defined update in that Pauli basis; both rows of one paired
    logical may not be zero.
    """

    names: tuple[str, ...]
    x: GF2Matrix
    z: GF2Matrix
    gauge_pair_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(self.names)
        if not names or any(
                not isinstance(name, str) or not name for name in names):
            raise ValueError(
                "record logical map names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("record logical map names must be unique")
        if not isinstance(self.x, GF2Matrix) or not isinstance(
                self.z, GF2Matrix):
            raise TypeError("record logical X/Z maps must be GF2Matrix values")
        if self.x.nrows != len(names) or self.z.nrows != len(names):
            raise ValueError(
                "record logical X/Z rows must match the protected basis")
        if self.x.ncols != self.z.ncols:
            raise ValueError(
                "record logical X/Z maps must use the same record basis")
        gauge_pair_indices = tuple(self.gauge_pair_indices)
        if any(not isinstance(index, int) or isinstance(index, bool) or
               index < 0 for index in gauge_pair_indices):
            raise TypeError(
                "record logical gauge-pair selectors must be nonnegative ints")
        if len(set(gauge_pair_indices)) != len(gauge_pair_indices):
            raise ValueError(
                "record logical gauge-pair selectors must be unique")
        if gauge_pair_indices and len(gauge_pair_indices) != len(names):
            raise ValueError(
                "record logical gauge-pair selectors must match names")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "gauge_pair_indices", gauge_pair_indices)


@dataclass(frozen=True, slots=True)
class MetaChecks:
    """Independent redundant-check relations for CSS X and Z records."""

    x: GF2Matrix | None = None
    z: GF2Matrix | None = None
    gauge: GF2Matrix | None = None

    def __post_init__(self) -> None:
        if self.x is not None and not isinstance(self.x, GF2Matrix):
            raise TypeError("MetaChecks.x must be a GF2Matrix")
        if self.z is not None and not isinstance(self.z, GF2Matrix):
            raise TypeError("MetaChecks.z must be a GF2Matrix")
        if self.gauge is not None and not isinstance(self.gauge, GF2Matrix):
            raise TypeError("MetaChecks.gauge must be a GF2Matrix")
        if self.x is None and self.z is None and self.gauge is None:
            raise ValueError(
                "MetaChecks requires an X, Z, or gauge relation matrix")


def _validate_metacheck_family(matrix,
                               checks,
                               family: str,
                               *,
                               binary_rows: bool = False) -> None:
    if matrix is None:
        return
    if matrix.ncols != len(checks):
        raise ValueError(
            f"{family} metachecks have width {matrix.ncols}, but the code "
            f"profile has {len(checks)} {family} checks")
    if matrix.rank != matrix.nrows:
        raise ValueError(
            f"{family} metacheck rows must be nonzero and independent")
    check_masks = tuple(
        sum((int(bit) << index)
            for index, bit in enumerate(row)) if binary_rows else sum(
                1 << index
                for index in row)
        for row in checks)
    for row_index, row in enumerate(matrix.rows):
        product = 0
        for selected, check in zip(row, check_masks):
            if selected:
                product ^= check
        if product:
            raise ValueError(
                f"{family} metacheck row {row_index} is not a relation among "
                "the declared effective checks")


class CodeProfile(ImmutableValue):
    __slots__ = (
        "name",
        "code",
        "distance",
        "effective_stabilizers",
        "effective_stabilizer_labels",
        "effective_stabilizer_indices",
        "decomposition",
        "effective_metachecks",
        "kept_from_effective",
        "metachecks",
        "gauge_measurements",
        "dynamic_phases",
        "transition_actions",
        "period_closure",
        "temporal_recovery",
        "temporal_recovery_targets",
        "record_logicals",
        "evidence",
        "metadata",
    )

    def __init__(
        self,
        code: "Code",
        *,
        name: str | None = None,
        distance: Distance | int | None = None,
        effective_stabilizers=None,
        effective_stabilizer_labels=None,
        decomposition: GF2Matrix | None = None,
        effective_metachecks: GF2Matrix | None = None,
        metachecks: MetaChecks | None = None,
        gauge_measurements: GaugeMeasurementMap | GF2Matrix | None = None,
        gauge_to_stabilizer: GF2Matrix | None = None,
        dynamic_phases: Iterable[MeasurementPhase] = (),
        period_closure: GF2Matrix | None = None,
        temporal_recovery: GF2Matrix | None = None,
        record_logicals: RecordLogicalMap | None = None,
        evidence: Iterable[Any] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.name = name or f"{code.name}_default_profile"
        self.distance = _distance(distance if distance is not None else code.d)

        inherited_effective_stabilizers = effective_stabilizers is None
        if inherited_effective_stabilizers:
            effective = code.declared_stabilizers
        elif isinstance(effective_stabilizers, GF2Matrix):
            effective = effective_stabilizers
        else:
            effective = GF2Matrix(
                tuple(
                    _declared_pauli_row(value, code.n)
                    for value in effective_stabilizers),
                ncols=2 * code.n,
            )
        if effective.ncols != 2 * code.n:
            raise ValueError(
                "effective stabilizers must have symplectic width 2n")
        if any(not any(row) for row in effective.rows):
            raise ValueError("effective stabilizers must be nonidentity")
        if effective.rank != code.stabilizer_basis.nrows:
            raise ValueError(
                "effective stabilizers must span the complete kept stabilizer group"
            )
        self.effective_stabilizers = effective
        default_effective_labels = (code.stabilizer_labels
                                    if inherited_effective_stabilizers else
                                    range(effective.nrows))
        (
            self.effective_stabilizer_labels,
            self.effective_stabilizer_indices,
        ) = _canonical_label_index(
            default_effective_labels if effective_stabilizer_labels is None else
            effective_stabilizer_labels,
            effective.nrows,
            what="effective_stabilizer_labels",
        )

        if decomposition is None:
            decomposition_rows = _coordinates_in_basis_many(
                effective.rows,
                code.stabilizer_basis.rows,
                _normalized=True,
            )
            decomposition = GF2Matrix._from_normalized_rows(
                decomposition_rows,
                ncols=code.stabilizer_basis.nrows,
            )
        if not isinstance(decomposition, GF2Matrix):
            raise TypeError("CodeProfile decomposition must be a GF2Matrix")
        if decomposition.nrows != effective.nrows or (
                decomposition.ncols != code.stabilizer_basis.nrows):
            raise ValueError(
                "effective-generator decomposition has incompatible dimensions")
        kept_encoded = tuple(
            _row_bits(row) for row in code.stabilizer_basis.rows)
        for index, (generator, coefficients) in enumerate(
                zip(effective.rows, decomposition.rows)):
            recovered = 0
            for selected, row in zip(coefficients, kept_encoded):
                if selected:
                    recovered ^= row
            if recovered != _row_bits(generator):
                raise ValueError(
                    f"effective-generator decomposition {index} is incorrect")
        self.decomposition = decomposition

        if effective_metachecks is None:
            effective_metacheck_rows = _row_relations(effective.rows,
                                                      ncols=2 * code.n,
                                                      _normalized=True)
            effective_metachecks = GF2Matrix._from_normalized_rows(
                effective_metacheck_rows,
                ncols=effective.nrows,
            )
        if not isinstance(effective_metachecks, GF2Matrix):
            raise TypeError("effective_metachecks must be a GF2Matrix")
        if effective_metachecks.ncols != effective.nrows:
            raise ValueError(
                "effective metachecks must index effective generators")
        if effective_metachecks.rank != effective_metachecks.nrows:
            raise ValueError(
                "effective metachecks must be nonzero and independent")
        effective_encoded = tuple(_row_bits(row) for row in effective.rows)
        decomposition_encoded = tuple(
            _row_bits(row) for row in decomposition.rows)
        for index, relation in enumerate(effective_metachecks.rows):
            effective_product = 0
            decomposition_product = 0
            for selected, effective_row, decomposition_row in zip(
                    relation, effective_encoded, decomposition_encoded):
                if selected:
                    effective_product ^= effective_row
                    decomposition_product ^= decomposition_row
            if effective_product:
                raise ValueError(
                    f"effective metacheck {index} is not a relation")
            if decomposition_product:
                raise ValueError(
                    f"effective metacheck {index} does not cancel in the kept basis"
                )
        expected_relation_count = effective.nrows - code.stabilizer_basis.nrows
        if effective_metachecks.nrows != expected_relation_count:
            raise ValueError(
                "effective metachecks must span every redundancy relation")
        self.effective_metachecks = effective_metachecks

        canonical_boundary_maps = _derive_boundary_maps(code, effective,
                                                        decomposition,
                                                        effective_metachecks)
        self.kept_from_effective = canonical_boundary_maps.kept_from_effective

        if metachecks == ():
            metachecks = None
        if metachecks is not None and not isinstance(metachecks, MetaChecks):
            raise TypeError(
                "CodeProfile.metachecks must be cudaq.logical.MetaChecks")
        if metachecks is not None:
            _validate_metacheck_family(metachecks.x, code.hx, "X")
            _validate_metacheck_family(metachecks.z, code.hz, "Z")
            for family, matrix, offset, width in (
                ("X", metachecks.x, 0, len(code.hx)),
                ("Z", metachecks.z, len(code.hx), len(code.hz)),
            ):
                if matrix is None:
                    continue
                for row in matrix.rows:
                    embedded = ((0,) * offset + row + (0,) *
                                (effective.nrows - offset - width))
                    if not _in_span(embedded, effective_metachecks.rows):
                        raise ValueError(
                            f"{family} metacheck is not present in the canonical "
                            "effective-generator relation space")
        self.metachecks = metachecks
        if isinstance(gauge_measurements, GF2Matrix):
            if not isinstance(gauge_to_stabilizer, GF2Matrix):
                raise TypeError(
                    "raw gauge_measurements require gauge_to_stabilizer=GF2Matrix"
                )
            gauge_measurements = GaugeMeasurementMap(gauge_measurements,
                                                     gauge_to_stabilizer)
        elif gauge_to_stabilizer is not None:
            raise TypeError(
                "gauge_to_stabilizer is only used with raw gauge measurements")
        if gauge_measurements is not None and not isinstance(
                gauge_measurements, GaugeMeasurementMap):
            raise TypeError("gauge_measurements must be a GaugeMeasurementMap")
        stabilizers, gauge_group = _code_symplectic_rows(code)
        if gauge_measurements is not None:
            if gauge_measurements.operators.ncols != 2 * code.n:
                raise ValueError(
                    "gauge operators must have symplectic width 2n")
            if any(not _in_span(row, gauge_group)
                   for row in gauge_measurements.operators.rows):
                raise ValueError(
                    "measured operator is outside the declared gauge group")
            recovered = gauge_measurements.recovered_stabilizers
            if any(not any(row) for row in recovered.rows):
                raise ValueError(
                    "gauge-to-stabilizer rows must recover nonidentity")
            if any(not _in_span(row, stabilizers) for row in recovered.rows):
                raise ValueError("gauge products must recover kept stabilizers")
            if metachecks is not None and metachecks.gauge is not None:
                _validate_metacheck_family(
                    metachecks.gauge,
                    gauge_measurements.operators.rows,
                    "gauge",
                    binary_rows=True,
                )
        elif metachecks is not None and metachecks.gauge is not None:
            raise ValueError("gauge metachecks require gauge measurements")
        self.gauge_measurements = gauge_measurements

        phases = tuple(dynamic_phases)
        if any(not isinstance(phase, MeasurementPhase) for phase in phases):
            raise TypeError(
                "dynamic_phases must contain MeasurementPhase values")
        if len({phase.name for phase in phases}) != len(phases):
            raise ValueError("dynamic phase names must be unique")
        if phases and any(not phase.input_epoch or not phase.output_epoch
                          for phase in phases):
            raise ValueError(
                "dynamic phases must declare input_epoch and output_epoch")
        if record_logicals is not None and not isinstance(
                record_logicals, RecordLogicalMap):
            raise TypeError("record_logicals must be a RecordLogicalMap")
        if record_logicals is not None and not phases:
            raise ValueError(
                "record_logicals require a periodic dynamic profile")
        if record_logicals is not None:
            if code.k:
                if len(record_logicals.names) != code.k:
                    raise ValueError(
                        "record logical pairs must equal the code's protected "
                        "logical-pair count")
                if record_logicals.gauge_pair_indices:
                    raise ValueError(
                        "stabilizer-code record logicals may not select gauge "
                        "pairs")
                expected_names = tuple(f"q{index}" for index in range(code.k))
                if record_logicals.names != expected_names:
                    raise ValueError(
                        "stabilizer-code record logical names must use the "
                        "canonical q0..q{k-1} basis order")
            else:
                if len(record_logicals.gauge_pair_indices) != len(
                        record_logicals.names):
                    raise ValueError(
                        "a k=0 dynamic profile must bind every record logical "
                        "to a distinct canonical gauge pair")
                if any(index >= code.r
                       for index in record_logicals.gauge_pair_indices):
                    raise ValueError(
                        "record logical gauge-pair selector is outside the "
                        "linked code's canonical gauge basis")
        protected_names = (record_logicals.names if record_logicals is not None
                           else tuple(f"q{index}" for index in range(code.k)))
        phase_record_count = sum(
            phase.measured_gauges.nrows for phase in phases)
        if (record_logicals is not None and
                record_logicals.x.ncols != phase_record_count):
            raise ValueError(
                "record logical maps must span every phase gauge record")
        transition_actions = []
        for phase in phases:
            if (phase.measured_gauges.ncols != 2 * code.n or
                    phase.instantaneous_stabilizers.ncols != 2 * code.n):
                raise ValueError("dynamic phase operators must have width 2n")
            if any(not any(row) or not _in_span(row, gauge_group)
                   for row in phase.measured_gauges.rows):
                raise ValueError(
                    "phase measurement must be a nonidentity member of the "
                    "gauge group")
            if (any(not any(row)
                    for row in phase.instantaneous_stabilizers.rows) or
                    any(not _in_span(row, gauge_group)
                        for row in phase.instantaneous_stabilizers.rows)):
                raise ValueError(
                    "phase instantaneous stabilizer lies outside the gauge group"
                )
            for left_index, left in enumerate(
                    phase.instantaneous_stabilizers.rows):
                for right in phase.instantaneous_stabilizers.rows[left_index +
                                                                  1:]:
                    if _symplectic_product(left, right, code.n):
                        raise ValueError(
                            "phase instantaneous stabilizers must commute")
            logical_map = phase.logical_map
            if set(logical_map) != set(protected_names):
                raise ValueError(
                    "phase logical_map must be total over the protected "
                    "periodic basis")
            mapped_names = tuple(logical_map.values())
            if (any(not isinstance(name, str) or name not in protected_names
                    for name in mapped_names) or
                    len(set(mapped_names)) != len(mapped_names)):
                raise ValueError(
                    "phase logical_map must be a bijection over the protected "
                    "periodic basis")
            if phase.logical_action is None:
                port_index = {
                    name: index for index, name in enumerate(protected_names)
                }
                width = 2 * len(protected_names)
                rows = []
                for offset in (0, len(protected_names)):
                    for name in protected_names:
                        row = [0] * width
                        row[offset + port_index[logical_map[name]]] = 1
                        rows.append(tuple(row))
                action = CliffordAction.from_symplectic(
                    matrix=rows,
                    phases=(0,) * width,
                    ports=protected_names,
                )
            else:
                action = phase.logical_action
                if action.ports != protected_names:
                    raise ValueError(
                        "phase logical_action ports must equal the protected "
                        "periodic basis in canonical order")
                if any(action.phases):
                    raise ValueError(
                        "signed phase logical actions are not representable in "
                        "the sign-free period closure")
            transition_actions.append(action)
            if phase.temporal_recovery is not None:
                if phase.temporal_recovery.ncols != phase.measured_gauges.nrows:
                    raise ValueError(
                        "phase recovery width must equal measurement count")
                if (phase.temporal_recovery @ phase.measured_gauges
                   ).rows != phase.instantaneous_stabilizers.rows:
                    raise ValueError("phase recovery does not produce its ISG")
        if phases:
            names = tuple(phase.name for phase in phases)
            inputs = tuple(phase.input_epoch for phase in phases)
            outputs = tuple(phase.output_epoch for phase in phases)
            if inputs != names:
                raise ValueError(
                    "dynamic phase names must equal their input epochs so row "
                    "and epoch identity cannot diverge")
            if outputs != names[1:] + names[:1]:
                raise ValueError(
                    "dynamic phases must form one ordered periodic epoch cycle")
            protected_count = len(protected_names)
            if not protected_count:
                raise ValueError(
                    "a periodic dynamic profile requires a nonempty protected "
                    "periodic basis")
            if code.k:
                protected_rows = (
                    *code.logical_x_basis.rows,
                    *code.logical_z_basis.rows,
                )
            else:
                protected_rows = (
                    *(code.gauge_x_basis.rows[index]
                      for index in record_logicals.gauge_pair_indices),
                    *(code.gauge_z_basis.rows[index]
                      for index in record_logicals.gauge_pair_indices),
                )
            expected_closure_width = 2 * protected_count
            if len(protected_rows) != expected_closure_width:
                raise ValueError(
                    "protected periodic representatives do not match the "
                    "bound paired basis")
            for left_index, left in enumerate(protected_rows):
                for right_index, right in enumerate(protected_rows):
                    expected = ((left_index < protected_count and
                                 right_index == left_index + protected_count) or
                                (right_index < protected_count and
                                 left_index == right_index + protected_count))
                    if bool(_symplectic_product(left, right,
                                                code.n)) != expected:
                        raise ValueError(
                            "protected periodic representatives must form a "
                            "canonical paired symplectic basis")
            initial_isg = _independent_rows(
                (*stabilizers, *phases[0].instantaneous_stabilizers.rows),
                ncols=2 * code.n,
            )
            if any(
                    _symplectic_product(logical, isg, code.n)
                    for logical in protected_rows
                    for isg in initial_isg):
                raise ValueError(
                    "protected periodic representatives must commute with the "
                    "initial instantaneous stabilizer group")
            quotient_basis = (*protected_rows, *initial_isg)
            if GF2Matrix(quotient_basis,
                         ncols=2 * code.n).rank != len(quotient_basis):
                raise ValueError(
                    "protected periodic representatives must be independent "
                    "modulo the initial instantaneous stabilizers")
            if record_logicals is None:
                record_updates = GF2Matrix(
                    ((0,) * phase_record_count,) * expected_closure_width,
                    ncols=phase_record_count,
                )
            else:
                record_updates = GF2Matrix(
                    (*record_logicals.x.rows, *record_logicals.z.rows),
                    ncols=phase_record_count,
                )
            transported_rows = protected_rows
            record_offset = 0
            for phase_index, phase in enumerate(phases):
                phase_width = phase.measured_gauges.nrows
                if any(
                        _symplectic_product(logical, gauge, code.n)
                        for logical in transported_rows
                        for gauge in phase.measured_gauges.rows):
                    raise ValueError(
                        "current protected periodic representatives must "
                        "commute with the active phase gauge measurements "
                        "before applying record-defined updates")
                phase_updates = GF2Matrix(
                    tuple(row[record_offset:record_offset + phase_width]
                          for row in record_updates.rows),
                    ncols=phase_width,
                )
                update_operators = phase_updates @ phase.measured_gauges
                transported_rows = tuple(
                    tuple(left ^ right
                          for left, right in zip(logical, update))
                    for logical, update in zip(transported_rows,
                                               update_operators.rows))
                record_offset += phase_width

                next_phase = phases[(phase_index + 1) % len(phases)]
                next_isg = _independent_rows(
                    (*stabilizers, *next_phase.instantaneous_stabilizers.rows),
                    ncols=2 * code.n,
                )
                for left_index, left in enumerate(transported_rows):
                    for right_index, right in enumerate(transported_rows):
                        expected = (
                            (left_index < protected_count and
                             right_index == left_index + protected_count) or
                            (right_index < protected_count and
                             left_index == right_index + protected_count))
                        if bool(_symplectic_product(left, right,
                                                    code.n)) != expected:
                            raise ValueError(
                                "record-defined logical transport must "
                                "preserve the canonical protected pairing at "
                                "every phase boundary")
                if any(
                        _symplectic_product(logical, isg, code.n)
                        for logical in transported_rows
                        for isg in next_isg):
                    raise ValueError(
                        "record-defined logical transport must commute with "
                        "the next instantaneous stabilizer group")
                next_quotient = (*next_isg, *transported_rows)
                if GF2Matrix(next_quotient,
                             ncols=2 * code.n).rank != len(next_quotient):
                    raise ValueError(
                        "record-defined logical transport must remain "
                        "independent modulo every instantaneous stabilizer "
                        "group")
            try:
                transported_coordinates = _coordinates_in_basis_many(
                    transported_rows, quotient_basis)
            except ValueError as exc:
                raise ValueError(
                    "record-defined logical transport does not close in the "
                    "initial protected/ISG quotient") from exc
            derived_closure = _validate_symplectic_closure(
                GF2Matrix(
                    tuple(row[:expected_closure_width]
                          for row in transported_coordinates),
                    ncols=expected_closure_width,
                ),
                what="derived period_closure",
            )
            action_product = GF2Matrix(
                tuple(
                    tuple(
                        int(row == column)
                        for column in range(expected_closure_width))
                    for row in range(expected_closure_width)),
                ncols=expected_closure_width,
            )
            for action in transition_actions:
                action_product = action_product @ GF2Matrix.from_rows(
                    action.matrix)
            if action_product != derived_closure:
                raise ValueError(
                    "phase logical actions contradict the exact physical "
                    "record/ISG transport derivation")
            if period_closure is not None:
                period_closure = _validate_symplectic_closure(
                    period_closure, what="period_closure")
                if period_closure != derived_closure:
                    raise ValueError(
                        "period_closure contradicts the exact physical "
                        "record/ISG transport derivation")
            period_closure = derived_closure
        elif period_closure is not None:
            raise ValueError(
                "period_closure requires a periodic dynamic profile")
        if temporal_recovery is not None:
            if not isinstance(temporal_recovery, GF2Matrix):
                raise TypeError("temporal_recovery must be a GF2Matrix")
            if temporal_recovery.ncols != phase_record_count:
                raise ValueError(
                    "temporal recovery width must span all phase records")
            phase_measurements = GF2Matrix(
                tuple(row for phase in phases
                      for row in phase.measured_gauges.rows),
                ncols=2 * code.n,
            )
            recovered = temporal_recovery @ phase_measurements
            if any(not any(row) or not _in_span(row, stabilizers)
                   for row in recovered.rows):
                raise ValueError(
                    "temporal recovery rows must recover nonidentity kept "
                    "stabilizers")
            kept_coordinates = GF2Matrix(
                _coordinates_in_basis_many(recovered.rows, stabilizers),
                ncols=len(stabilizers),
            )
            temporal_recovery_targets = kept_coordinates
        else:
            temporal_recovery_targets = None
        self.dynamic_phases = phases
        self.transition_actions = tuple(transition_actions)
        self.period_closure = period_closure
        self.temporal_recovery = temporal_recovery
        self.temporal_recovery_targets = temporal_recovery_targets
        self.record_logicals = record_logicals

        self.evidence = tuple(
            _deep_freeze(item, what=f"CodeProfile.evidence[{index}]")
            for index, item in enumerate(evidence))
        self.metadata = _deep_freeze(dict(metadata or {}),
                                     what="CodeProfile.metadata")
        self._seal()

    def _boundary_maps(self) -> _BoundaryMaps:
        """Derive the complete operational maps from canonical profile facts."""

        return _derive_boundary_maps(
            self.code,
            self.effective_stabilizers,
            self.decomposition,
            self.effective_metachecks,
        )

    def decode_boundary(self, pauli):
        """Decode a physical Pauli into effective syndrome and logical bits."""

        return self._boundary_maps().decode(pauli)

    def encode_boundary(self, effective_syndrome, logical_x=(), logical_z=()):
        """Construct a canonical physical representative of boundary data."""

        return self._boundary_maps().encode(effective_syndrome, logical_x,
                                            logical_z)

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)


@dataclass(frozen=True, slots=True)
class EncodingEpochSchema:
    """Reusable finite state machine for an encoding's dynamic phases."""

    name: str
    phases: tuple[str, ...]
    initial: str
    transitions: tuple[tuple[str, str], ...] = ()
    logical_maps: Mapping[str, Any] | None = None
    is_periodic: bool = False
    closure: GF2Matrix | None = None

    def __post_init__(self) -> None:
        phases = tuple(
            str(getattr(phase, "name", phase)) for phase in self.phases)
        if not self.name or not phases or any(not phase for phase in phases):
            raise ValueError("encoding epoch schema requires a name and phases")
        if len(set(phases)) != len(phases):
            raise ValueError("encoding epoch phases must be unique")
        if self.initial not in phases:
            raise ValueError(
                "encoding epoch initial phase must belong to phases")
        transitions = tuple(
            (str(source), str(target)) for source, target in self.transitions)
        if any(source not in phases or target not in phases
               for source, target in transitions):
            raise ValueError(
                "encoding epoch transitions must reference declared phases")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "logical_maps",
                           _deep_freeze(dict(self.logical_maps or {})))
        if self.is_periodic:
            if not transitions:
                raise ValueError(
                    "periodic epoch schema requires explicit closure transitions"
                )
            if self.closure is None:
                raise ValueError(
                    "periodic epoch schema requires a closure matrix")
            object.__setattr__(
                self,
                "closure",
                _validate_symplectic_closure(self.closure,
                                             what="encoding epoch closure"),
            )
        elif self.closure is not None:
            raise ValueError(
                "encoding epoch closure requires a periodic schema")

    @classmethod
    def static(cls, *, name="static_epoch") -> "EncodingEpochSchema":
        return cls(name=name, phases=("initial",), initial="initial")

    @classmethod
    def gauge_dynamic(
            cls,
            *,
            fixed_stabilizers=(),
            gauge_basis=(),
            name="gauge_dynamic_epoch",
    ) -> "EncodingEpochSchema":
        """A two-phase schema for gauge-mediated dynamics.

        ``gauge_free`` is the ordinary operating phase; ``gauge_fixed``
        represents an interval where declared gauge operators have been
        measured/fixed. The kept stabilizer group is phase-independent;
        only the gauge embedding state changes, so both logical maps are
        identity. The declared bases ride along as descriptive facts.
        """
        maps = {
            "gauge_free":
                "identity",
            "gauge_fixed":
                "identity",
            "fixed_stabilizers":
                tuple(str(item) for item in tuple(fixed_stabilizers)),
            "gauge_basis":
                tuple(str(item) for item in tuple(gauge_basis)),
        }
        return cls(
            name=name,
            phases=("gauge_free", "gauge_fixed"),
            initial="gauge_free",
            transitions=(
                ("gauge_free", "gauge_fixed"),
                ("gauge_fixed", "gauge_free"),
            ),
            logical_maps=maps,
        )

    @classmethod
    def periodic(
        cls,
        *,
        phases,
        logical_maps=None,
        closure=None,
        name="periodic_epoch",
    ) -> "EncodingEpochSchema":
        names = tuple(str(getattr(phase, "name", phase)) for phase in phases)
        if not names:
            raise ValueError(
                "periodic epoch schema requires at least one phase")
        transitions = tuple((phase, names[(index + 1) % len(names)])
                            for index, phase in enumerate(names))
        return cls(
            name=name,
            phases=names,
            initial=names[0],
            transitions=transitions,
            logical_maps=logical_maps,
            is_periodic=True,
            closure=closure,
        )


@dataclass(frozen=True, slots=True)
class EncodingEpoch:
    """One immutable phase instance used to qualify a patch SSA type."""

    name: str
    encoding: "Encoding"
    schema: EncodingEpochSchema
    phase: str
    index: int = 0

    def __post_init__(self) -> None:
        if self.phase not in self.schema.phases:
            raise ValueError(
                "encoding epoch phase is not declared by its schema")
        if not isinstance(self.index, int) or isinstance(
                self.index, bool) or self.index < 0:
            raise TypeError(
                "encoding epoch index must be a nonnegative Python int")
