# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Pinnacle generalized-bicycle QEC architecture recipes.

The five published GB code rows compose their code, encoding, reusable gadget
definitions, same-block Webster--Smith--Cohen measurement lowering, direct PBC
injection, and certified repeat-until-success rotation synthesis into immutable
:class:`devices.QECArchitecture` values.

These recipes do not install process-global defaults. The module also exposes
the paper's magic engines as scheduled compact P3 factory models with explicit
timing and footprint evidence, not as gate-level distillation circuits. A
gate-level factory realization and circuit-distance claim remain explicit
follow-on work. The WSC body emits
its modified-base, chi, and gamma checks, but uses a serial bare ancilla and
therefore retains unknown circuit distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from inspect import Parameter, Signature
import math
from types import MappingProxyType

import cudaq.logical.algebra as algebra
import cudaq.logical.analysis.evidence as evidence
import cudaq.logical.codes as codes
import cudaq.logical.devices as devices
import cudaq.logical.gadgets as gadgets
import cudaq.logical.ops as ops
import cudaq.logical.protocols as protocols
import cudaq.logical.qec as qec
import cudaq.logical.std as standard
import cudaq.logical.types as types
from cudaq.logical.programs.decorators import objective as _objective

from ..std import LogicalActionRef, LogicalInstrumentRef

from . import _rus, _wsc

__all__ = (
    "PinnacleProcessingBlock",
    "RUSReadoutModel",
    "magic_engine",
    "processing_block",
    "gb30",
    "gb62",
    "gb126",
    "gb254",
    "gb510",
    "for_code",
)


class RUSReadoutModel(Enum):
    """Architecture-owned realization of the BRS ancilla measurement.

    ``EXPLICIT_WSC`` retains the constructive serial bare-ancilla reference
    circuit. ``PAPER_LOGICAL_CYCLE`` retains the same bounded BRS protocol and
    T-state demand but uses the native logical-product measurement assumed by
    the Pinnacle resource model.
    """

    EXPLICIT_WSC = "explicit_wsc"
    PAPER_LOGICAL_CYCLE = "paper_logical_cycle"


def _rus_readout_model(value) -> RUSReadoutModel:
    if not isinstance(value, RUSReadoutModel):
        raise TypeError("rus_readout_model must be a Pinnacle RUSReadoutModel")
    return value


@dataclass(frozen=True, slots=True)
class _CliffordSupport:
    """Exact support status for one searched global Clifford realization."""

    supported: bool
    operation: str
    reason: str
    candidates_checked: int
    stabilizer_preserving_candidates: int
    canonical_candidates: int
    candidate_family: str
    objective_action: algebra.CliffordAction


def _in_reduced_span(row: int, basis: tuple[int, ...]) -> bool:
    pivots = _reduced_span_pivots(basis)
    while row:
        pivot = row.bit_length() - 1
        pivot_row = pivots.get(pivot)
        if pivot_row is None:
            return False
        row ^= pivot_row
    return True


@lru_cache(maxsize=32)
def _reduced_span_pivots(basis: tuple[int, ...]):
    """Index a reduced GF(2) basis by pivot for sparse exact reduction."""

    return MappingProxyType({
        pivot_row.bit_length() - 1: pivot_row
        for pivot_row in basis
        if pivot_row
    })


def _permute_bits(row: int, permutation: tuple[int, ...]) -> int:
    output = 0
    while row:
        bit = row & -row
        qubit = bit.bit_length() - 1
        output |= 1 << permutation[qubit]
        row ^= bit
    return output


def _candidate_permutations(ell: int) -> tuple[tuple[int, ...], ...]:
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


def _fold_masks(row: int, permutation: tuple[int, ...]) -> tuple[int, int, int]:
    """Return the exact ``(x, z, phase)`` fold image without allocations."""

    x = z = phase = 0
    while row:
        bit = row & -row
        qubit = bit.bit_length() - 1
        row ^= bit
        image_z = 1 << permutation[qubit]
        # This is the exact PauliGroupElement multiplication rule specialized
        # to the ordered factors X_q Z_permutation(q).  Accumulating masks and
        # the phase directly avoids allocating one immutable group element for
        # every selected qubit in every audited candidate.
        phase += int(permutation[qubit] == qubit)
        phase += 2 * (z & bit).bit_count()
        x ^= bit
        z ^= image_z
    return x, z, phase & 3


def _fold_image(row: int, permutation: tuple[int, ...]) -> _wsc._LabeledPauli:
    return _wsc._LabeledPauli.from_masks(*_fold_masks(row, permutation))


def _canonical_global_action(kind: str,
                             logicals: int) -> algebra.CliffordAction:
    if kind == "h":
        images = tuple((0, 1 << index, 0) for index in range(logicals)) + tuple(
            (1 << index, 0, 0) for index in range(logicals))
    elif kind == "s":
        images = tuple(
            (1 << index, 1 << index, 0) for index in range(logicals)) + tuple(
                (0, 1 << index, 0) for index in range(logicals))
    else:
        raise ValueError("canonical global action supports h or s")
    return algebra.CliffordAction.from_images(
        images,
        ports=tuple(range(logicals)),
        evidence=("canonical-global",),
    )


def _audit_global_clifford(code, ell: int, kind: str) -> _CliffordSupport:
    """Audit Adam's declared GB fold family against a canonical H or S."""

    if kind not in ("h", "s"):
        raise ValueError("global Clifford audit supports h or s")
    hx = tuple(sum(1 << qubit for qubit in row) for row in code.hx)
    hz = tuple(sum(1 << qubit for qubit in row) for row in code.hz)
    lx = tuple(sum(1 << qubit for qubit in row) for row in code.lx)
    lz = tuple(sum(1 << qubit for qubit in row) for row in code.lz)
    hx_basis = _rref(hx)
    hz_basis = _rref(hz)
    candidates = _candidate_permutations(ell)
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
                if (z_coordinates != 1 << logical or
                        x_coordinates != 1 << logical):
                    exact = False
                    break
            canonical += int(exact)
            continue

        if any(permutation[permutation[index]] != index
               for index in range(code.n)):
            continue
        preserved = True
        for row in hx:
            image_x, image_z, image_phase = _fold_masks(row, permutation)
            if not (_in_reduced_span(image_x, hx_basis) and
                    _in_reduced_span(image_z, hz_basis) and image_phase == 0):
                preserved = False
                break
        if not preserved:
            continue
        stabilizer_preserving += 1
        exact = True
        for logical in range(code.k):
            image_x, image_z, image_phase = _fold_masks(lx[logical],
                                                        permutation)
            difference_x = image_x ^ lx[logical]
            difference_z = image_z ^ lz[logical]
            difference_phase = (image_phase + 1 + 2 *
                                (image_z & lx[logical]).bit_count()) & 3
            if not (_in_reduced_span(difference_x, hx_basis) and
                    _in_reduced_span(difference_z, hz_basis) and
                    difference_phase == 0):
                exact = False
                break
        canonical += int(exact)

    operation = kind.upper()
    return _CliffordSupport(
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


def _global_preparation_objective(code, logical_state, flip, name: str):

    def intent():
        values = []
        for _ in range(code.k):
            value = ops.prepare(state=logical_state)
            values.append(flip(value))
        return tuple(values)

    result_annotation = tuple[tuple([types.logical_qubit] * code.k)]
    intent.__name__ = intent.__qualname__ = f"{name}_objective"
    intent.__signature__ = Signature((), return_annotation=result_annotation)
    intent.__annotations__ = {"return": result_annotation}
    return _objective(intent, name=intent.__name__)


def _global_cnot_objective(code, name: str):
    controls = tuple(f"control{index}" for index in range(code.k))
    targets = tuple(f"target{index}" for index in range(code.k))
    names = controls + targets

    def intent(*values):
        values = list(values)
        for logical in range(code.k):
            values[logical], values[code.k + logical] = ops.cx(
                values[logical], values[code.k + logical])
        return tuple(values)

    result_annotation = tuple[tuple([types.logical_qubit] * (2 * code.k))]
    intent.__name__ = intent.__qualname__ = f"{name}_objective"
    intent.__signature__ = Signature(
        tuple(
            Parameter(
                item,
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=types.logical_qubit,
            ) for item in names),
        return_annotation=result_annotation,
    )
    intent.__annotations__ = {
        **{
            item: types.logical_qubit for item in names
        },
        "return": result_annotation,
    }
    return _objective(intent, name=intent.__name__)


@dataclass(frozen=True, slots=True)
class PinnacleGBInstance:
    """One published Pinnacle generalized-bicycle processing-block row."""

    name: str
    ell: int
    a: tuple[int, ...]
    b: tuple[int, ...]
    k: int
    distance: int
    code_block_qubits: int
    gadget_qubits: int
    bridge_qubits: int
    processing_block_qubits: int

    @property
    def n(self) -> int:
        return 2 * self.ell

    @property
    def logical_cycle_rounds(self) -> int:
        return self.distance + 2

    @property
    def report_provenance(self):
        """Evidence for the published parameter, cycle, and resource row."""

        return evidence.citation("arXiv:2602.11457v2 Table I")


@dataclass(frozen=True, slots=True, init=False)
class PinnacleProcessingBlock:
    """Typed P3 footprint for one published Pinnacle processing block.

    Construction is library-controlled so only a cited Table-I row can carry
    published-footprint provenance.  Obtain values with
    :func:`processing_block`; generic admitted codes deliberately have no
    published physical footprint.
    """

    preset: str
    logical_capacity: int
    code_block_qubits: int
    gadget_qubits: int
    bridge_qubits: int
    physical_qubits: int

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        raise TypeError("PinnacleProcessingBlock is library-controlled; use "
                        "pinnacle.processing_block(architecture)")

    @property
    def report_provenance(self):
        return evidence.citation("arXiv:2602.11457v2 Table I")


def _processing_block_value(
        instance: PinnacleGBInstance) -> PinnacleProcessingBlock:
    block = object.__new__(PinnacleProcessingBlock)
    for name, value in (
        ("preset", instance.name),
        ("logical_capacity", instance.k),
        ("code_block_qubits", instance.code_block_qubits),
        ("gadget_qubits", instance.gadget_qubits),
        ("bridge_qubits", instance.bridge_qubits),
        ("physical_qubits", instance.processing_block_qubits),
    ):
        object.__setattr__(block, name, value)
    return block


_PINNACLE_GB_INSTANCES = MappingProxyType({
    "gb30":
        PinnacleGBInstance("gb30", 15, (0, 6, 13), (0, 1, 4), 8, 4, 60, 13, 7,
                           140),
    "gb62":
        PinnacleGBInstance("gb62", 31, (0, 6, 15), (0, 5, 7), 10, 6, 124, 19,
                           11, 244),
    "gb126":
        PinnacleGBInstance("gb126", 63, (0, 4, 37), (0, 29, 49), 12, 10, 252,
                           31, 19, 452),
    "gb254":
        PinnacleGBInstance(
            "gb254",
            127,
            (0, 32, 100),
            (0, 28, 49),
            14,
            16,
            508,
            57,
            31,
            860,
        ),
    "gb510":
        PinnacleGBInstance(
            "gb510",
            255,
            (0, 39, 55),
            (0, 70, 127),
            16,
            24,
            1020,
            99,
            51,
            1620,
        ),
})

_BY_DISTANCE = {row.distance: row for row in _PINNACLE_GB_INSTANCES.values()}


@dataclass(frozen=True, slots=True)
class _MagicEngine:
    """One published Pinnacle T-state compact factory operating point.

    The value is a production and footprint model, not a
    gate-level distillation circuit. Its fields are the paper operating point
    needed by device construction and P3 scheduling. The computed
    output infidelity uses the paper's approximate formula and its rounded
    input values; ``published_target_output_infidelity`` records the nominal
    operating point assigned to the row rather than clamping that
    approximation. Instances are internal so arbitrary values cannot inherit
    published provenance.
    """

    name: str
    physical_error_rate: float
    published_target_output_infidelity: float
    gb_distance: int
    input_error: float
    ancilla_distance: int
    ancilla_logical_error_per_cycle: float
    postselection_rounds: int
    physical_qubits: int
    reject_rate: float
    reaction_code_cycles: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("magic-engine name must be nonempty")
        for field in (
                "physical_error_rate",
                "published_target_output_infidelity",
        ):
            value = getattr(self, field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"magic-engine {field} must be numeric")
            if not 0.0 < value <= 1.0:
                raise ValueError(f"magic-engine {field} must lie in (0, 1]")
        for field in (
                "input_error",
                "ancilla_logical_error_per_cycle",
                "reject_rate",
        ):
            value = getattr(self, field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"magic-engine {field} must be numeric")
            if not 0.0 <= value < 1.0:
                raise ValueError(f"magic-engine {field} must lie in [0, 1)")
        for field in (
                "gb_distance",
                "ancilla_distance",
                "postselection_rounds",
                "physical_qubits",
                "reaction_code_cycles",
        ):
            value = getattr(self, field)
            if (not isinstance(value, int) or isinstance(value, bool) or
                    value <= 0):
                raise TypeError(f"magic-engine {field} must be a positive int")

    @property
    def rotation_error(self) -> float:
        return self.input_error + (self.ancilla_distance +
                                   1) * self.ancilla_logical_error_per_cycle

    @property
    def postselection_error(self) -> float:
        return self.physical_error_rate**self.postselection_rounds

    @property
    def output_infidelity(self) -> float:
        """Eq. (5) first-line interpretation used by this factory model.

        The paper's expanded line and numerical example are inconsistent with
        this expression. The named producer metadata makes this fixed modeling
        choice replay-visible; it is not presented as a source-independent
        bound.
        """

        rotation = self.rotation_error
        postselection = self.postselection_error
        return 35.0 * rotation**3 + 6.0 * rotation * postselection**2

    @property
    def cycles_per_attempt(self) -> int:
        """Paper service period for one magic-engine attempt.

        Equation (11) gives the internal distillation duration, while the
        engine can hand an output to the GB processing fabric only on a
        logical-cycle boundary.  The externally observable attempt cadence is
        therefore the slower of those two periods.
        """

        return max(self.gb_distance + 2, self.distillation_cycles)

    @property
    def distillation_cycles(self) -> int:
        """Internal Eq. (11) duration before GB-cycle synchronization."""

        distance = self.ancilla_distance
        rounds = self.postselection_rounds
        reaction = self.reaction_code_cycles
        return max(
            2 * distance + 4 * rounds,
            reaction + 4 * rounds,
            distance + reaction + 3 * rounds,
        )

    @property
    def acceptance_probability(self) -> float:
        return max(0.0, 1.0 - self.reject_rate)

    @property
    def producer(self) -> protocols.ProtocolDefinition:
        """Typed analytical producer used by ``logical.add_factory``."""

        return _magic_engine_producer(self)

    @property
    def provenance(self):
        return evidence.citation(
            "arXiv:2602.11457v2 Section V.B.2, Eqs. (4)-(11)")


_MAGIC_ENGINES = (
    _MagicEngine(
        "pinnacle_me_p1e_4_pout1e_9",
        1.0e-4,
        1.0e-9,
        10,
        1.0e-4,
        1,
        1.0e-4,
        1,
        592,
        19.0e-4,
    ),
    _MagicEngine(
        "pinnacle_me_p1e_4_pout1e_11",
        1.0e-4,
        1.0e-11,
        16,
        5.0e-5,
        5,
        5.0e-7,
        1,
        1807,
        0.02,
    ),
    _MagicEngine(
        "pinnacle_me_p1e_3_pout1e_9",
        1.0e-3,
        1.0e-9,
        24,
        1.0e-4,
        7,
        2.0e-5,
        2,
        4410,
        0.10,
    ),
    _MagicEngine(
        "pinnacle_me_p1e_3_pout1e_11",
        1.0e-3,
        1.0e-11,
        24,
        5.0e-5,
        9,
        2.0e-6,
        2,
        5430,
        0.10,
    ),
)


@lru_cache(maxsize=None)
def _magic_engine_producer(
        engine: _MagicEngine) -> protocols.ProtocolDefinition:
    """Materialize the explicitly analytical producer for one table row."""

    def produce_t():
        return ops.produce(
            standard.T_STATE,
            protocol=engine.name,
        )

    produce_t.__name__ = produce_t.__qualname__ = f"{engine.name}_produce_t"
    produce_t.__annotations__ = {
        "return": types.resource[standard.T_STATE],
    }
    producer = protocols.ProtocolDefinition(
        produce_t,
        implements=standard.produce(standard.T_STATE),
        name=produce_t.__name__,
        metadata={
            "production_model": engine.name,
            "factory_mode": "scheduled_macro",
            "produces": standard.T_STATE.name,
            "physical_error_rate": engine.physical_error_rate,
            "cycles_per_attempt": engine.cycles_per_attempt,
            "distillation_cycles": engine.distillation_cycles,
            "pipeline_depth": 1,
            "published_target_output_infidelity":
                (engine.published_target_output_infidelity),
            "acceptance_probability": engine.acceptance_probability,
            "reject_probability": engine.reject_rate,
            "output_infidelity": engine.output_infidelity,
            "output_error_model":
                ("eq5_first_line_rotation_times_postselection_squared"),
            "acceptance_model": "paper_quoted_operating_point",
            "source_formula_status": "paper_internal_inconsistency_disclosed",
            "physical_qubits": engine.physical_qubits,
            "provenance": ("arXiv:2602.11457v2 Section V.B.2, Eqs. (4)-(11)"),
        },
    )
    producer._seal()
    return producer


def magic_engine(
    *,
    p_phys: float,
    target_output_infidelity: float,
) -> _MagicEngine:
    """Select the smallest row meeting the named Eq.-(5) interpretation.

    The row's nominal target and the library's explicit first-line Eq.-(5)
    model must both meet the requested value. The paper contains conflicting
    expanded/numerical formulas, so callers should treat ``output_infidelity``
    as model-dependent evidence rather than an unambiguous published bound.
    """

    p_phys = float(p_phys)
    target_output_infidelity = float(target_output_infidelity)
    if not 0.0 < p_phys <= 1.0:
        raise ValueError("p_phys must lie in (0, 1]")
    if not 0.0 < target_output_infidelity <= 1.0:
        raise ValueError("target_output_infidelity must lie in (0, 1]")
    candidates = tuple(engine for engine in _MAGIC_ENGINES if math.isclose(
        engine.physical_error_rate,
        p_phys,
        rel_tol=0.0,
        abs_tol=1.0e-16,
    ) and engine.published_target_output_infidelity <= target_output_infidelity
                       and engine.output_infidelity <= target_output_infidelity)
    if not candidates:
        raise ValueError(
            f"no published Pinnacle magic engine for p_phys={p_phys:g} and "
            f"target output infidelity <= {target_output_infidelity:g}")
    return min(candidates, key=lambda engine: engine.physical_qubits)


_PUBLISHED_GB_SEEDS = MappingProxyType({
    ell: MappingProxyType(seeds) for ell, seeds in {
        31: {
            "x0": ((1, 6, 8, 10), (11, 26)),
            "z0": ((3, 12, 18, 19), (11, 18)),
            "x1": ((16, 23), (0, 15, 16, 22)),
            "z1": ((0, 16), (1, 3, 5, 10)),
        },
        63: {
            "x0": ((7, 12, 36, 41, 56), (1, 27, 31, 38, 61)),
            "z0": ((5, 15, 28, 35, 45, 61), (1, 11, 54, 57)),
            "x1": ((9, 19, 26, 29), (5, 15, 22, 38, 48, 55)),
            "z1": ((2, 25, 32, 36, 62), (7, 22, 27, 51, 56)),
        },
        127: {
            "x0": (
                (28, 47, 55, 75, 103, 114, 124),
                (4, 14, 15, 23, 50, 77, 83, 109, 123),
            ),
            "z0": (
                (1, 24, 33, 51, 60, 65, 107, 119, 124),
                (7, 8, 36, 85, 106, 114, 124),
            ),
            "x1": (
                (3, 31, 32, 42, 52, 60, 81),
                (6, 15, 38, 42, 47, 59, 101, 106, 115),
            ),
            "z1": (
                (0, 8, 9, 19, 27, 41, 67, 73, 100),
                (26, 36, 47, 75, 95, 103, 122),
            ),
        },
        255: {
            "x0": (
                (18, 31, 35, 36, 91, 126, 146, 163, 164, 180, 196, 216, 233,
                 253),
                (48, 52, 87, 101, 103, 106, 107, 125, 140, 156, 179, 211),
            ),
            "z0": (
                (38, 54, 57, 93, 112, 148, 164, 185, 197, 203, 213, 238, 240,
                 252),
                (18, 55, 59, 73, 129, 130, 142, 182, 187, 199, 244, 252),
            ),
            "x1": (
                (6, 27, 35, 80, 92, 97, 137, 149, 150, 206, 220, 224),
                (27, 39, 41, 66, 76, 82, 94, 115, 131, 167, 186, 222, 225, 241),
            ),
            "z1": (
                (10, 11, 14, 16, 30, 65, 69, 161, 193, 216, 232, 247),
                (26, 81, 82, 86, 99, 119, 139, 156, 176, 192, 208, 209, 226,
                 246),
            ),
        },
    }.items()
})


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


def _quotient_basis(centralizer: tuple[int, ...], stabilizers: tuple[int, ...],
                    count: int) -> tuple[int, ...]:
    span = list(_rref(stabilizers))
    output = []
    for row in centralizer:
        if not _in_span(row, tuple(span)):
            span.append(row)
            output.append(row)
            if len(output) == count:
                return tuple(output)
    raise ValueError(
        f"homology quotient supplied {len(output)}, expected {count}")


def _solve(rows: tuple[int, ...], rhs: tuple[int, ...], columns: int) -> int:
    equations = [row | (bit << columns) for row, bit in zip(rows, rhs)]
    pivot_row = 0
    pivots = []
    for column in range(columns):
        found = next(
            (index for index in range(pivot_row, len(equations))
             if (equations[index] >> column) & 1),
            None,
        )
        if found is None:
            continue
        equations[pivot_row], equations[found] = (
            equations[found],
            equations[pivot_row],
        )
        for index in range(len(equations)):
            if index != pivot_row and (equations[index] >> column) & 1:
                equations[index] ^= equations[pivot_row]
        pivots.append(column)
        pivot_row += 1
    mask = (1 << columns) - 1
    if any((equation & mask) == 0 and ((equation >> columns) & 1)
           for equation in equations):
        raise ValueError("logical pairing matrix is singular")
    solution = 0
    for equation, pivot in zip(equations[:pivot_row], pivots):
        if (equation >> columns) & 1:
            solution |= 1 << pivot
    return solution


def _canonicalize_lz(lx: tuple[int, ...], lz: tuple[int,
                                                    ...]) -> tuple[int, ...]:
    k = len(lx)
    pairing = tuple(
        sum(((x & z).bit_count() & 1) << column
            for column, z in enumerate(lz))
        for x in lx)
    output = []
    for logical in range(k):
        coefficients = _solve(
            pairing,
            tuple(int(index == logical) for index in range(k)),
            k,
        )
        row = 0
        for index, old in enumerate(lz):
            if (coefficients >> index) & 1:
                row ^= old
        output.append(row)
    return tuple(output)


def _gb_matrices(instance: PinnacleGBInstance):
    ell = instance.ell
    hx = tuple(
        tuple(sorted({(row + shift) % ell
                      for shift in instance.a})) +
        tuple(ell + qubit
              for qubit in sorted({(row + shift) % ell
                                   for shift in instance.b}))
        for row in range(ell))
    hz = tuple(
        tuple(sorted({(row - shift) % ell
                      for shift in instance.b})) +
        tuple(ell + qubit
              for qubit in sorted({(row - shift) % ell
                                   for shift in instance.a}))
        for row in range(ell))
    return hx, hz


def _pack(rows) -> tuple[int, ...]:
    return tuple(sum(1 << qubit for qubit in row) for row in rows)


def _unpack(row: int) -> tuple[int, ...]:
    return tuple(
        index for index in range(row.bit_length()) if (row >> index) & 1)


def _seed_row(ell: int, seed) -> int:
    left, right = seed
    return sum(1 << qubit for qubit in left) | sum(
        1 << (ell + qubit) for qubit in right)


def _orbit(row: int, ell: int) -> tuple[int, ...]:
    output = []
    for shift in range(ell):
        shifted = 0
        for qubit in range(2 * ell):
            if (row >> qubit) & 1:
                sector, index = divmod(qubit, ell)
                shifted |= 1 << (sector * ell + (index + shift) % ell)
        output.append(shifted)
    return tuple(output)


def _select_orbits(families, stabilizers, count_per_family):
    span = list(_rref(stabilizers))
    selected = []
    for family in families:
        before = len(selected)
        for row in family:
            if not _in_span(row, tuple(span)):
                span.append(row)
                selected.append(row)
                if len(selected) - before == count_per_family:
                    break
        if len(selected) - before != count_per_family:
            raise ValueError(
                "published seed orbit does not span its logical sector")
    return tuple(selected)


def _pinnacle_gb_instance(value: str | int) -> PinnacleGBInstance:
    """Resolve a preset by canonical name or by its published distance."""

    if isinstance(value, str):
        try:
            return _PINNACLE_GB_INSTANCES[value]
        except KeyError as exc:
            raise ValueError(f"unknown Pinnacle GB preset {value!r}; expected "
                             f"{tuple(_PINNACLE_GB_INSTANCES)}") from exc
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            return _BY_DISTANCE[value]
        except KeyError as exc:
            raise ValueError(
                f"unknown Pinnacle GB distance {value!r}; expected "
                f"{tuple(sorted(_BY_DISTANCE))}") from exc
    raise TypeError("Pinnacle GB preset must be a name or distance int")


@lru_cache(maxsize=None)
def _pinnacle_gb_by_name(name: str) -> codes.CSSCode:
    """Construct one canonical preset after public alias normalization."""

    instance = _PINNACLE_GB_INSTANCES[name]
    hx, hz = _gb_matrices(instance)
    hx_bits = _pack(hx)
    hz_bits = _pack(hz)
    actual_k = instance.n - _rank(hx_bits) - _rank(hz_bits)
    if actual_k != instance.k:
        raise ValueError(
            f"{instance.name} matrix rank gives k={actual_k}, expected {instance.k}"
        )

    seed_evidence = None
    if instance.ell == 15:
        lx_bits = _quotient_basis(_nullspace(hz_bits, instance.n), hx_bits,
                                  instance.k)
        lz_bits = _quotient_basis(_nullspace(hx_bits, instance.n), hz_bits,
                                  instance.k)
    else:
        seeds = _PUBLISHED_GB_SEEDS[instance.ell]
        packed = {
            name: _seed_row(instance.ell, seed) for name, seed in seeds.items()
        }
        x_orbits = (_orbit(packed["x0"],
                           instance.ell), _orbit(packed["x1"], instance.ell))
        z_orbits = (_orbit(packed["z0"],
                           instance.ell), _orbit(packed["z1"], instance.ell))
        half = instance.k // 2
        lx_bits = _select_orbits(x_orbits, hx_bits, half)
        lz_bits = _select_orbits(z_orbits, hz_bits, half)
        seed_evidence = MappingProxyType({
            "source":
                "arXiv:2511.15989v1 Appendix A",
            "x_sector_dimensions": (
                _rank(hx_bits + x_orbits[0]) - _rank(hx_bits),
                _rank(hx_bits + x_orbits[0] + x_orbits[1]) -
                _rank(hx_bits + x_orbits[0]),
            ),
        })

    lz_bits = _canonicalize_lz(lx_bits, lz_bits)
    metadata = {
        "family": "pinnacle_generalized_bicycle",
        "ell": instance.ell,
        "a": instance.a,
        "b": instance.b,
    }
    if seed_evidence is not None:
        metadata["seed_evidence"] = seed_evidence
    return codes.CSSCode(
        name=f"pinnacle_{instance.name}",
        n=instance.n,
        k=instance.k,
        d=codes.Distance.claimed(
            instance.distance,
            provenance=evidence.citation("arXiv:2602.11457v2 Table I"),
        ),
        block=codes.CSSBlock(data=instance.n, sx=instance.ell, sz=instance.ell),
        hx=hx,
        hz=hz,
        lx=tuple(_unpack(row) for row in lx_bits),
        lz=tuple(_unpack(row) for row in lz_bits),
        metadata=metadata,
    )


def _pinnacle_gb(value: str | int) -> codes.CSSCode:
    """Return one cached native Mark III Pinnacle generalized-bicycle code."""

    return _pinnacle_gb_by_name(_pinnacle_gb_instance(value).name)


class _CSSDefinitionBuilder:
    """Private constructor for one fixed high-rate CSS definition set."""

    def __init__(self, preset: str | int):
        self.instance = _pinnacle_gb_instance(preset)
        self.code = _pinnacle_gb(self.instance.name)
        self.encoding = self.code.default_encoding

    def prepare_zero(self):
        return gadgets.prepare_zero(self.encoding,
                                    name=f"{self.code.name}_prepare_zero")

    def prepare_plus(self):
        return gadgets.prepare_plus(self.encoding,
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
        operation = ops.z if plus_basis else ops.x
        intent = _global_preparation_objective(
            code,
            types.plus if plus_basis else types.zero,
            operation,
            f"{code.name}_prepare_{suffix}",
        )

        def realization(block):
            block = base(block)
            return operation(block.data[flip_support])

        realization.__name__ = realization.__qualname__ = intent.name
        realization.__annotations__ = {
            "block": types.patch[encoding],
            "return": types.patch[encoding],
        }
        return gadgets.gadget(
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
            for _ in range(rounds):
                block, _ = ops.extract_syndrome(block)
            return block

        realization.__name__ = realization.__qualname__ = (
            f"{code.name}_memory_r{rounds}")
        realization.__annotations__ = {
            "block": types.patch[encoding],
            "return": types.patch[encoding],
        }
        return gadgets.gadget(
            realization,
            implements=standard.idle,
            name=realization.__name__,
            metadata={"syndrome_rounds": rounds},
        )

    def h_evidence(self) -> _CliffordSupport:
        return _audit_global_clifford(self.code, self.instance.ell, "h")

    def s_evidence(self) -> _CliffordSupport:
        return _audit_global_clifford(self.code, self.instance.ell, "s")

    def transversal_cx(self) -> gadgets.GadgetDefinition:
        encoding = self.encoding
        code = self.code
        intent = _global_cnot_objective(code, f"{code.name}_transversal_cx")

        def realization(control, target):
            return ops.cx(control.data, target.data)

        realization.__name__ = realization.__qualname__ = f"{code.name}_transversal_cx"
        realization.__annotations__ = {
            "control": types.patch[encoding],
            "target": types.patch[encoding],
            "return": tuple[types.patch[encoding], types.patch[encoding]],
        }
        return gadgets.gadget(
            realization,
            implements=intent,
            logical_ports={
                **{
                    f"control{index}": f"control.{encoding.ports[index].name}" for index in range(code.k)
                },
                **{
                    f"target{index}": f"target.{encoding.ports[index].name}" for index in range(code.k)
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
    ) -> _wsc.WSCMeasurementPlan:
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
        ) = _wsc._build_wsc(self.code, normalized_terms, rounds, sign=sign)
        sign_key = "p" if sign > 0 else "m"
        name = (f"{self.code.name}_wsc_" +
                "_".join(f"{pauli.lower()}{logical}"
                         for logical, pauli in normalized_terms) +
                f"_{sign_key}_r{rounds}")
        definition = _wsc._measurement_gadget(
            self.code,
            self.encoding,
            normalized_terms,
            requested=requested,
            base_checks=base,
            merged_checks=merged,
            chi_checks=chi,
            incidence=incidence,
            kappa_count=kappa_count,
            rounds=rounds,
            evidence=evidence,
            name=name,
            sign=sign,
        )
        return _wsc.WSCMeasurementPlan(
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
    logical_cycle_rounds: int | None


def _require_high_rate_css(value) -> tuple[codes.Code, codes.Encoding]:
    """Normalize and fail closed unless ``value`` is completely CSS-authored."""

    if isinstance(value, codes.Encoding):
        code = value.code
        encoding = value
    elif isinstance(value, codes.Code):
        code = value
        encoding = value.default_encoding
    else:
        raise TypeError(
            "high-rate CSS definitions require a codes.Code or codes.Encoding")

    css_check_rank = _rank(_pack(code.hx)) + _rank(_pack(code.hz))
    if (code.r != 0 or code.stabilizers or code.gauges or code.gx or code.gz or
            len(code.lx) != code.k or len(code.lz) != code.k or
            css_check_rank != code.stabilizer_basis.nrows):
        raise TypeError(
            "high-rate CSS definitions require a stabilizer code represented "
            "entirely by hx/hz checks and lx/lz logicals")

    partitions = code.block.partitions
    if (partitions.get("data") != code.n or
            partitions.get("sx", 0) != len(code.hx) or
            partitions.get("sz", 0) != len(code.hz)):
        raise ValueError("high-rate CSS definitions require carrier partitions "
                         "data=n, sx=len(hx), and sz=len(hz)")
    return code, encoding


class _HighRateCSSBuilder(_CSSDefinitionBuilder):
    """Private Mark III definition builder for an arbitrary native CSS code.

    General CSS codes get preparation, repeated memory, transversal CX, and
    exact within-block WSC product measurement. Global H/S are exposed only
    when the code is a Pinnacle GB instance whose declared fold family can be
    exhaustively audited; otherwise the result is precisely unsupported.
    """

    def __init__(self, value, *, logical_cycle_rounds: int | None = None):
        code, encoding = _require_high_rate_css(value)
        if (logical_cycle_rounds is not None and
            (not isinstance(logical_cycle_rounds, int) or isinstance(
                logical_cycle_rounds, bool) or logical_cycle_rounds <= 0)):
            raise ValueError(
                "logical_cycle_rounds must be a positive int when provided")
        self.code = code
        self.encoding = encoding
        self.instance = _GenericCSSInstance(logical_cycle_rounds)

    def _global_support(self, kind: str) -> _CliffordSupport:
        ell = self.code.metadata.get("ell")
        if (self.code.metadata.get("family") == "pinnacle_generalized_bicycle"
                and isinstance(ell, int) and not isinstance(ell, bool)):
            return _audit_global_clifford(self.code, ell, kind)
        return _CliffordSupport(
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

    def h_evidence(self) -> _CliffordSupport:
        return self._global_support("h")

    def s_evidence(self) -> _CliffordSupport:
        return self._global_support("s")


def _pinnacle_instance_for_code(code: codes.Code) -> PinnacleGBInstance | None:
    """Match a code to one report row using only code-algebra facts."""

    if code.metadata.get("family") != "pinnacle_generalized_bicycle":
        return None
    ell = code.metadata.get("ell")
    a = code.metadata.get("a")
    b = code.metadata.get("b")
    if (not isinstance(ell, int) or isinstance(ell, bool) or
            not isinstance(a, tuple) or not isinstance(b, tuple)):
        return None
    key = (
        ell,
        a,
        b,
        code.n,
        code.k,
        code.d.value,
    )
    for instance in _PINNACLE_GB_INSTANCES.values():
        if key == (instance.ell, instance.a, instance.b, instance.n, instance.k,
                   instance.distance):
            return instance
    return None


def _default_cycle_rounds(code: codes.Code) -> int:
    instance = _pinnacle_instance_for_code(code)
    if instance is not None:
        return instance.logical_cycle_rounds
    return qec.rounds.from_code_distance.resolve(code)


def _resolve_rounds(code: codes.Code,
                    rounds: int | qec.rounds.RoundPolicy | None) -> int:
    if rounds is None:
        return _default_cycle_rounds(code)
    if isinstance(rounds, qec.rounds.RoundPolicy):
        return rounds.resolve(code)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise TypeError(
            "rounds must be a positive int or qec.rounds.RoundPolicy")
    return rounds


def _formal_terms(code: codes.Code,
                  product: types.PauliProduct) -> tuple[tuple[int, str], ...]:
    if not isinstance(product, types.PauliProduct):
        raise TypeError("WSC measurement expects a types.PauliProduct")
    if product.identities:
        raise ValueError(
            "WSC measurement does not accept identity-covered ports")
    terms = []
    for factor in product.factors:
        logical = factor.operand
        if not isinstance(logical, int) or isinstance(logical, bool):
            raise TypeError(
                "WSC Pauli factors must use integer logical-port operands")
        if not 0 <= logical < code.k:
            raise ValueError(
                f"code {code.name} has no protected logical port {logical}")
        terms.append((logical, factor.pauli))
    return tuple(terms)


def _builder(value) -> _HighRateCSSBuilder:
    return _HighRateCSSBuilder(value)


def _wsc_measurement(
    value,
    product: types.PauliProduct,
    *,
    rounds: int | qec.rounds.RoundPolicy | None = None,
) -> _wsc.WSCMeasurementBundle:
    """Build an exact within-block WSC measurement for ``product``.

    Formal integer operands select protected ports of ``value``. For example,
    ``types.X(0) @ types.Y(2)`` requests ``X`` on port 0 times ``Y`` on port 2.
    """

    builder = _builder(value)
    code = builder.code
    resolved_rounds = _resolve_rounds(code, rounds)
    plan = builder.build_wsc_measurement(
        _formal_terms(code, product),
        rounds=resolved_rounds,
        sign=product.sign,
    )
    return _wsc.WSCMeasurementBundle(
        product=product,
        realization=plan.gadget,
        analysis=_wsc.measurement_profile(plan,
                                          name=f"{plan.gadget.name}_analysis"),
        evidence=plan.evidence,
        rounds=resolved_rounds,
        data_code=code,
        diagnostics=plan,
    )


def _canonical_mpp_parameters(site) -> tuple[int, int, int]:
    expected = {"x_mask", "z_mask", "sign"}
    actual = set(site.parameters)
    if actual != expected:
        raise ValueError(
            "Pinnacle WSC lowering requires exactly x_mask, z_mask, and sign; "
            f"missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}")
    values = tuple(
        site.parameters[name] for name in ("x_mask", "z_mask", "sign"))
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in values):
        raise TypeError("MPP masks and sign must be non-bool integers")
    return values


def _site_product(site, context) -> types.PauliProduct:
    """Map an MPP site's ordered P0 masks through packed P1 placements."""

    if site.objective_family != "pauli_product_measurement":
        raise ValueError("Pinnacle WSC lowering accepts only P0 MPP sites")
    x_mask, z_mask, sign = _canonical_mpp_parameters(site)
    if sign not in (-1, 1):
        raise ValueError("MPP sign must be +1 or -1")
    width = len(context.placements)
    support = x_mask | z_mask
    if support == 0 or support >> width:
        raise ValueError(
            "MPP masks must describe the complete action-site boundary")

    blocks = {binding.block for binding in context.placements}
    if None in blocks or len(blocks) != 1:
        raise ValueError(
            "Pinnacle WSC lowering requires one explicitly packed QEC block")

    logical_indices = tuple(
        binding.logical_index for binding in context.placements)
    if any(index is None for index in logical_indices):
        raise ValueError("packed WSC operands require logical-index witnesses")
    if len(set(logical_indices)) != len(logical_indices):
        raise ValueError("packed WSC logical-index witnesses must be injective")

    product = None
    for position, logical_index in enumerate(logical_indices):
        assert logical_index is not None
        if not 0 <= logical_index < context.code.k:
            raise ValueError(
                "packed WSC logical index is outside the selected code")
        x_bit = (x_mask >> position) & 1
        z_bit = (z_mask >> position) & 1
        if not x_bit and not z_bit:
            raise ValueError(
                "Pinnacle WSC lowering does not accept identity operands")
        factory = types.Y if x_bit and z_bit else types.X if x_bit else types.Z
        factor = factory(logical_index)
        product = factor if product is None else product @ factor
    assert product is not None
    return product if sign > 0 else -product


def _make_wsc_lowering(
    encoding: codes.Encoding,
    *,
    default_rounds: int,
) -> qec.QECLowering:
    """Create the P1-to-P2 compiler for one packed CSS block."""

    def compile_site(site, context):
        product = _site_product(site, context)
        rounds = context.policy.get("rounds", default_rounds)
        resolved_rounds = _resolve_rounds(context.code, rounds)
        measurement = _wsc_measurement(context.encoding,
                                       product,
                                       rounds=resolved_rounds)
        annotation = types.patch[context.encoding]

        def generated(block):
            return measurement.realization(block, analysis=measurement.analysis)

        product_key = (f"x{int(site.parameters['x_mask']):x}_"
                       f"z{int(site.parameters['z_mask']):x}_"
                       f"{'m' if product.sign < 0 else 'p'}")
        generated.__name__ = f"{context.lowering.name}_{product_key}"
        generated.__qualname__ = generated.__name__
        generated.__module__ = context.lowering.provider.__module__
        result_annotation = tuple[annotation, bool]
        generated.__signature__ = Signature(
            (Parameter(
                "block",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation,
            ),),
            return_annotation=result_annotation,
        )
        hints = {"block": annotation, "return": result_annotation}
        protocol = protocols.ProtocolDefinition(
            generated,
            implements=standard.mpp,
            name=generated.__name__,
            type_hints=hints,
            metadata={
                "compiler": "cudaq.logical.architectures.pinnacle.wsc",
                "rounds": resolved_rounds,
            },
        )
        return qec.GeneratedQECArtifact(
            protocol,
            {
                "x_mask":
                    int(site.parameters["x_mask"]),
                "z_mask":
                    int(site.parameters["z_mask"]),
                "sign":
                    int(site.parameters.get("sign", 1)),
                "rounds":
                    resolved_rounds,
                "kappa_qubits":
                    measurement.kappa_qubits,
                "merged_checks":
                    measurement.merged_check_count,
                "code_distance_status":
                    measurement.evidence.code_distance.status,
                "circuit_distance_status":
                    measurement.evidence.circuit_distance.status,
            },
        )

    compile_site.__name__ = f"compile_{encoding.name}_wsc_mpp"
    compile_site.__qualname__ = compile_site.__name__
    return qec.QECLowering(
        compile_site,
        objective_family="pauli_product_measurement",
        objective=standard.mpp,
        codes=(encoding,),
        plugin="cudaq.logical.architectures.pinnacle_wsc",
        version="1.0.0",
        name=f"{encoding.name}_wsc_mpp",
        policy_schema={"rounds": "positive int"},
        metadata={
            "construction": "Webster-Smith-Cohen merged-code measurement",
            "ownership": "one packed QEC block",
        },
    )


def _quarter_turn_rotation_plan(site, context):
    """Derive one selected exact quarter-turn protected-port product."""

    if site.objective_family != "pauli_product_rotation":
        raise ValueError(
            "Pinnacle quarter-turn lowering accepts only Pauli-product rotations"
        )
    required = {"x_mask", "z_mask", "sign", "angle"}
    missing = required - set(site.parameters)
    if missing:
        raise ValueError(
            "Pinnacle quarter-turn lowering requires a static product and angle; "
            f"missing={sorted(missing)!r}")
    x_mask, z_mask, sign = (
        site.parameters[name] for name in ("x_mask", "z_mask", "sign"))
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in (x_mask, z_mask, sign)):
        raise TypeError("quarter-turn masks and sign must be integers")
    if sign not in (-1, 1):
        raise ValueError("quarter-turn Pauli-product sign must be +1 or -1")
    exact = qec.product_rotation.signed_pi_fraction(site.parameters)
    if exact is not None:
        numerator, denominator = exact
        if denominator <= 0 or (4 * numerator) % denominator:
            raise ValueError(
                "selected Pinnacle exact angle is not on the pi/4 grid")
        quarter_turns = (4 * numerator) // denominator
    else:
        angle = site.parameters["angle"]
        if not isinstance(angle, (int, float)) or isinstance(angle, bool):
            raise TypeError("quarter-turn angle must be numeric")
        signed_quarters = sign * float(angle) / (math.pi / 4.0)
        quarter_turns = round(signed_quarters)
        if not math.isclose(
                signed_quarters, quarter_turns, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "selected Pinnacle quarter-turn is not on the pi/4 grid")

    # R_P(theta + 2*pi) and R_P(theta) differ only by global phase.  Reduce
    # before constructing strategy metadata so arbitrarily large exact
    # rational numerators never pass through a lossy float conversion.
    quarter_turns = ((quarter_turns + 4) % 8) - 4

    placements = tuple(context.placements)
    width = len(placements)
    support = x_mask | z_mask
    if width == 0 or support == 0 or support >> width:
        raise ValueError("PBC masks must fit a nonempty action-site boundary")
    if any(binding.binding_kind != "local" for binding in placements):
        raise ValueError(
            "Pinnacle PBC lowering currently requires local placement")
    if any(binding.logical_index is None for binding in placements):
        raise ValueError(
            "Pinnacle PBC operands require protected logical-index witnesses")

    block_keys = tuple(
        dict.fromkeys(
            binding.block or binding.placement for binding in placements))
    block_positions = {name: index for index, name in enumerate(block_keys)}
    terms = []
    for position, binding in enumerate(placements):
        x_bit = (x_mask >> position) & 1
        z_bit = (z_mask >> position) & 1
        if not x_bit and not z_bit:
            continue
        logical_index = binding.logical_index
        assert logical_index is not None
        if not 0 <= logical_index < context.code.k:
            raise ValueError(
                "Pinnacle PBC logical index is outside the selected encoding")
        terms.append((
            block_positions[binding.block or binding.placement],
            logical_index,
            "Y" if x_bit and z_bit else "X" if x_bit else "Z",
        ))
    return block_keys, tuple(terms), int(quarter_turns)


def _make_clifford_rotation_strategy(encoding: codes.Encoding):
    """Create exact no-resource Clifford product rotations for Pinnacle."""

    def compile_site(site, context):
        block_keys, terms, quarter_turns = _quarter_turn_rotation_plan(
            site, context)
        if quarter_turns % 2:
            raise ValueError(
                "Pinnacle Clifford strategy requires a pi/2 grid angle")
        annotation = types.patch[context.encoding]
        owner_count = len(block_keys)
        block_parameters = tuple(
            Parameter(
                f"block{index}",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation,
            ) for index in range(owner_count))
        result_annotation = tuple[tuple([annotation] * owner_count)]
        canonical_quarters = ((quarter_turns + 4) % 8) - 4

        def generated(*blocks):
            if canonical_quarters == 0:
                return tuple(blocks)
            product = None
            for block_position, logical_index, pauli in terms:
                factor = {
                    "X": types.X,
                    "Y": types.Y,
                    "Z": types.Z
                }[pauli](blocks[block_position][logical_index])
                product = factor if product is None else product @ factor
            assert product is not None
            if canonical_quarters < 0:
                product = -product
            involved = tuple(sorted({term[0] for term in terms}))
            updated = ops.rotate(
                product,
                angle=abs(canonical_quarters) * math.pi / 4.0,
            )
            output = list(blocks)
            for block_position, successor in zip(involved, updated):
                output[block_position] = successor
            return tuple(output)

        suffix = f"q{quarter_turns % 8}"
        generated.__name__ = f"{context.lowering.name}_{suffix}_clifford"
        generated.__qualname__ = generated.__name__
        generated.__module__ = context.lowering.provider.__module__
        generated.__signature__ = Signature(
            block_parameters,
            return_annotation=result_annotation,
        )
        hints = {
            **{
                parameter.name: annotation for parameter in block_parameters
            },
            "return": result_annotation,
        }
        gadget = gadgets.GadgetDefinition(
            generated,
            implements=standard.pauli_rotation,
            name=generated.__name__,
            type_hints=hints,
            metadata={
                "compiler": "cudaq.logical.architectures.pinnacle",
                "construction": "exact encoded Clifford product rotation",
                "quarter_turns": quarter_turns,
                "data_owners": owner_count,
            },
        )
        return qec.GeneratedQECArtifact(
            gadget,
            {
                "angle_pi_numer": quarter_turns,
                "angle_pi_denom": 4,
                "quarter_turns": quarter_turns,
                "data_owners": owner_count,
                "resource_count": 0,
            },
        )

    compile_site.__name__ = f"compile_{encoding.name}_clifford_rotation_strategy"
    compile_site.__qualname__ = compile_site.__name__
    return compile_site


def _make_pbc_rotation_strategy(encoding: codes.Encoding):
    """Create the one-T exact odd-quarter-turn Pinnacle strategy."""

    def compile_site(site, context):
        block_keys, terms, quarter_turns = _quarter_turn_rotation_plan(
            site, context)
        if not quarter_turns % 2:
            raise ValueError(
                "Pinnacle T injection requires an odd pi/4 grid angle")
        canonical_quarters = ((quarter_turns + 4) % 8) - 4
        injection_quarter = 1 if canonical_quarters > 0 else -1
        clifford_quarters = canonical_quarters - injection_quarter
        annotation = types.patch[context.encoding]
        resource_annotation = types.resource[standard.T_STATE]
        owner_count = len(block_keys)

        def inject(*values):
            blocks = values[:-1]
            state = values[-1]
            product = None
            for block_position, logical_index, pauli in terms:
                factor = {
                    "X": types.X,
                    "Y": types.Y,
                    "Z": types.Z
                }[pauli](blocks[block_position][logical_index])
                product = factor if product is None else product @ factor
            assert product is not None
            if injection_quarter < 0:
                product = -product
            involved = tuple(sorted({term[0] for term in terms}))
            updated = ops.resource_rotate(
                state,
                product,
                angle=math.pi / 4.0,
            )
            output = list(blocks)
            for block_position, successor in zip(involved, updated):
                output[block_position] = successor
            if clifford_quarters:
                product = None
                for block_position, logical_index, pauli in terms:
                    factor = {
                        "X": types.X,
                        "Y": types.Y,
                        "Z": types.Z
                    }[pauli](output[block_position][logical_index])
                    product = factor if product is None else product @ factor
                assert product is not None
                if clifford_quarters < 0:
                    product = -product
                updated = ops.rotate(
                    product,
                    angle=abs(clifford_quarters) * math.pi / 4.0,
                )
                for block_position, successor in zip(involved, updated):
                    output[block_position] = successor
            return tuple(output)

        product_key = (
            f"x{int(site.parameters['x_mask']):x}_"
            f"z{int(site.parameters['z_mask']):x}_q{quarter_turns % 8}")
        block_parameters = tuple(
            Parameter(
                f"block{index}",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation,
            ) for index in range(owner_count))
        result_annotation = tuple[tuple([annotation] * owner_count)]
        inject.__name__ = f"{context.lowering.name}_{product_key}_inject"
        inject.__qualname__ = inject.__name__
        inject.__module__ = context.lowering.provider.__module__
        inject.__signature__ = Signature(
            (
                *block_parameters,
                Parameter(
                    "state",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=resource_annotation,
                ),
            ),
            return_annotation=result_annotation,
        )
        injection = gadgets.GadgetDefinition(
            inject,
            implements=standard.pauli_rotation,
            name=inject.__name__,
            type_hints={
                **{
                    parameter.name: annotation for parameter in block_parameters
                },
                "state": resource_annotation,
                "return": result_annotation,
            },
            metadata={
                "construction": "typed T-state resource rotation",
                "resource": standard.T_STATE.name,
                "quarter_turns": quarter_turns,
            },
        )

        def generated(*blocks):
            state = ops.request_many(standard.T_STATE, count=1)[0]
            return injection(*blocks, state)

        generated.__name__ = f"{context.lowering.name}_{product_key}"
        generated.__qualname__ = generated.__name__
        generated.__module__ = context.lowering.provider.__module__
        generated.__signature__ = Signature(
            block_parameters,
            return_annotation=result_annotation,
        )
        hints = {
            **{
                parameter.name: annotation for parameter in block_parameters
            },
            "return": result_annotation,
        }
        protocol = protocols.ProtocolDefinition(
            generated,
            implements=standard.pauli_rotation,
            name=generated.__name__,
            type_hints=hints,
            metadata={
                "compiler": "cudaq.logical.architectures.pinnacle",
                "construction": "typed T-state product rotation",
                "resource": standard.T_STATE.name,
                "quarter_turns": quarter_turns,
                "data_owners": owner_count,
            },
        )
        return qec.GeneratedQECArtifact(
            protocol,
            {
                "x_mask": int(site.parameters["x_mask"]),
                "z_mask": int(site.parameters["z_mask"]),
                "quarter_turns": quarter_turns,
                "angle_pi_numer": quarter_turns,
                "angle_pi_denom": 4,
                "data_owners": owner_count,
                "resource_kind": standard.T_STATE.name,
                "resource_count": 1,
            },
        )

    compile_site.__name__ = f"compile_{encoding.name}_pbc_rotation_strategy"
    compile_site.__qualname__ = compile_site.__name__
    return compile_site


def _rus_rotation_plan(site, context):
    """Map one arbitrary RPP site to protected GB ports and a signed angle."""

    if site.objective_family != "pauli_product_rotation":
        raise ValueError("Pinnacle RUS synthesis accepts only RPP action sites")
    required = {"x_mask", "z_mask", "sign", "angle"}
    missing = required - set(site.parameters)
    if missing:
        raise ValueError(
            "Pinnacle RUS synthesis requires a static product and angle; "
            f"missing={sorted(missing)!r}")
    x_mask = site.parameters["x_mask"]
    z_mask = site.parameters["z_mask"]
    sign = site.parameters["sign"]
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in (x_mask, z_mask, sign)):
        raise TypeError("RUS masks and sign must be non-bool integers")
    if sign not in (-1, 1):
        raise ValueError("RUS Pauli-product sign must be +1 or -1")
    angle = site.parameters["angle"]
    precision = site.parameters.get("precision",
                                    context.policy.get("rpp_precision"))
    if (not isinstance(angle, (int, float)) or isinstance(angle, bool) or
            not math.isfinite(float(angle))):
        raise TypeError("Pinnacle RUS angle must be a finite number")
    if (not isinstance(precision,
                       (int, float)) or isinstance(precision, bool) or
            not math.isfinite(float(precision)) or float(precision) <= 0.0):
        raise TypeError(
            "Pinnacle RUS precision must be supplied by the action site or "
            "policy['rpp_precision'] as a finite positive number")

    placements = tuple(context.placements)
    width = len(placements)
    support = x_mask | z_mask
    if width == 0 or support == 0 or support >> width:
        raise ValueError("RUS masks must fit a nonempty action-site boundary")
    if any(binding.binding_kind != "local" for binding in placements):
        raise ValueError(
            "Pinnacle RUS synthesis currently requires local placement")
    if any(binding.logical_index is None for binding in placements):
        raise ValueError(
            "Pinnacle RUS operands require logical-index witnesses")

    block_keys = tuple(
        dict.fromkeys(
            binding.block or binding.placement for binding in placements))
    block_positions = {name: index for index, name in enumerate(block_keys)}
    terms = []
    for position, binding in enumerate(placements):
        x_bit = (x_mask >> position) & 1
        z_bit = (z_mask >> position) & 1
        if not x_bit and not z_bit:
            continue
        logical_index = binding.logical_index
        assert logical_index is not None
        if not 0 <= logical_index < context.code.k:
            raise ValueError(
                "Pinnacle RUS logical index is outside the selected encoding")
        terms.append((
            block_positions[binding.block or binding.placement],
            logical_index,
            "Y" if x_bit and z_bit else "X" if x_bit else "Z",
        ))
    spaces = {binding.space for binding in placements}
    if len(spaces) != 1:
        raise ValueError("Pinnacle RUS operands must share one logical region")
    return (
        block_keys,
        tuple(terms),
        sign * float(angle),
        float(precision),
        spaces.pop(),
    )


def _rus_scratch_region(context, logical_region: str) -> devices.QECRegion:
    bindings = tuple(binding for binding in context.device.logical_to_qec
                     if binding.logical_region.name == logical_region and
                     binding.architecture is not None)
    if len(bindings) != 1:
        raise ValueError(
            "Pinnacle RUS synthesis requires exactly one architecture binding "
            f"for logical region {logical_region!r}")
    matches = tuple(region for region in bindings[0].auxiliary_regions
                    if region.encoding is context.encoding and
                    region.metadata.get("owner") == "pinnacle_rus")
    if len(matches) != 1:
        names = tuple(region.name for region in matches)
        raise ValueError(
            "Pinnacle RUS synthesis requires one unambiguous architecture-owned "
            f"scratch region; got {names!r}")
    return matches[0]


def _rus_pauli_correction(context, terms, *, name: str):
    """Construct the exact encoded Pauli that restores a failed RUS round."""

    encoding = context.encoding
    annotation = types.patch[encoding]
    owner_count = len(
        tuple(
            dict.fromkeys(binding.block or binding.placement
                          for binding in context.placements)))
    x_masks = [0] * owner_count
    z_masks = [0] * owner_count
    for owner, logical, pauli in terms:
        if pauli in ("X", "Y"):
            for qubit in context.code.lx[logical]:
                x_masks[owner] ^= 1 << qubit
        if pauli in ("Z", "Y"):
            for qubit in context.code.lz[logical]:
                z_masks[owner] ^= 1 << qubit

    def correction(*blocks):
        output = list(blocks)
        for owner, mask in enumerate(x_masks):
            support = tuple(
                qubit for qubit in range(context.code.n) if (mask >> qubit) & 1)
            if support:
                output[owner] = ops.x(output[owner].data[support])
        for owner, mask in enumerate(z_masks):
            support = tuple(
                qubit for qubit in range(context.code.n) if (mask >> qubit) & 1)
            if support:
                output[owner] = ops.z(output[owner].data[support])
        return tuple(output)

    parameters = tuple(
        Parameter(
            f"block{index}",
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation,
        ) for index in range(owner_count))
    result_annotation = tuple[tuple([annotation] * owner_count)]
    correction.__name__ = correction.__qualname__ = name
    correction.__module__ = context.lowering.provider.__module__
    correction.__signature__ = Signature(
        parameters,
        return_annotation=result_annotation,
    )
    return gadgets.GadgetDefinition(
        correction,
        # This helper is only the encoded Pauli that normalizes the failure
        # branch of one RUS attempt.  It must not advertise the unconditional
        # rotation objective owned by the enclosing bounded protocol.
        implements=LogicalActionRef(
            "pinnacle_rus_failure_correction",
            owner_count,
        ),
        name=name,
        type_hints={
            **{
                parameter.name: annotation for parameter in parameters
            },
            "return": result_annotation,
        },
        metadata={
            "construction": "encoded Pauli failure correction",
            "x_masks": tuple(x_masks),
            "z_masks": tuple(z_masks),
        },
    )


@lru_cache(maxsize=16)
def _rus_scratch_measurement(
    encoding: codes.Encoding,
    rounds: int,
) -> _wsc.WSCMeasurementBundle:
    """Return the immutable WSC profile shared by RUS sites for one encoding.

    Keeping this bounded cache outside the provider closure avoids making the
    large symbolic profile part of every device's architecture-closure walk.
    The encoding and resolved round count are the complete inputs to the fixed
    ``-Z(0)`` RUS scratch measurement.
    """

    return _wsc_measurement(encoding, -types.Z(0), rounds=rounds)


def _make_rus_rotation_strategy(
    encoding: codes.Encoding,
    *,
    default_rounds: int,
    readout_model: RUSReadoutModel,
    max_attempts: int = 32,
):
    """Create a concrete BRS attempt plus named bounded-retry RPP strategy."""

    readout_model = _rus_readout_model(readout_model)

    def compile_site(site, context):
        block_keys, terms, angle, precision, logical_region = _rus_rotation_plan(
            site, context)
        scratch_region = _rus_scratch_region(context, logical_region)
        design = _rus.synthesize(angle, precision)
        annotation = types.patch[context.encoding]
        resource_annotation = types.resource[standard.T_STATE]
        owner_count = len(block_keys)
        # One attempt is a private ideal instrument over the selected Pauli
        # product's logical operands.  Its Boolean outcome distinguishes the
        # requested-rotation branch from the corrected identity branch; only
        # the enclosing bounded protocol implements unconditional
        # ``pauli_rotation``.
        attempt_objective = LogicalInstrumentRef(
            "pinnacle_rus_attempt",
            arity=len(terms),
            result_arity=1,
        )
        involved_data = tuple(sorted({owner for owner, _, _ in terms}))
        prepare_scratch = gadgets.prepare_zero(
            context.encoding,
            name=f"{context.lowering.name}_{design.sha256[:12]}_scratch_prepare",
        )
        if readout_model is RUSReadoutModel.EXPLICIT_WSC:
            scratch_measurement = _rus_scratch_measurement(
                context.encoding,
                default_rounds,
            )
            scratch_readout = scratch_measurement.realization
            # Profile success rows are rejection predicates; retry accepts
            # only when every selected row is false. The returned WSC bit is
            # the BRS success bit, so its complementary affine row is the
            # attempt-failure predicate.
            rejection_rows = (gadgets.SuccessPredicate(
                _wsc.measurement_result_parity(scratch_measurement.diagnostics)
                ^ True),)
        else:
            # The paper treats the BRS ancilla readout as one logical Pauli
            # measurement.  Keep it as a typed encoded MPP rather than
            # substituting a scalar duration: P3 still projects and schedules
            # the selected native instrument over the exact logical support.
            scratch_readout = gadgets.measure_z(
                context.encoding,
                logical=0,
                preserve_block=True,
                name=(f"{context.lowering.name}_paper_cycle_rus_readout"),
            )
            rejection_rows = (gadgets.SuccessPredicate(
                gadgets.ProfileParity.from_value(
                    scratch_readout.record("mpp0.outcome")) ^ True),)
        attempt_analysis = gadgets.GadgetProfile(
            scratch_readout,
            success=rejection_rows,
            name=(f"{context.lowering.name}_{design.sha256[:12]}_rus_success"),
            metadata={
                "success_probability":
                    design.success_probability,
                "success_probability_evidence":
                    f"synthesis:sha256:{design.sha256}",
                "readout_model":
                    readout_model.value,
            },
        )
        correction = _rus_pauli_correction(
            context,
            terms,
            name=f"{context.lowering.name}_{design.sha256[:12]}_failure_pauli",
        )
        block_parameters = tuple(
            Parameter(
                f"block{index}",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation,
            ) for index in range(owner_count))
        attempt_result = tuple[tuple([annotation] * owner_count + [bool])]

        def lifted_unitary(*values):
            output = list(values[:owner_count])
            scratch = values[owner_count]
            states = values[owner_count + 1:]
            state_cursor = 0
            for column in design.columns:
                product = None
                if column.data:
                    for owner, logical, pauli in terms:
                        factor = {
                            "X": types.X,
                            "Y": types.Y,
                            "Z": types.Z
                        }[pauli](output[owner][logical])
                        product = factor if product is None else product @ factor
                if column.ancilla is not None:
                    factor = {
                        "X": types.X,
                        "Y": types.Y,
                        "Z": types.Z
                    }[column.ancilla](scratch[0])
                    product = factor if product is None else product @ factor
                assert product is not None
                if column.consumes_t_state:
                    if column.quarter_turns < 0:
                        product = -product
                    successors = ops.resource_rotate(
                        states[state_cursor],
                        product,
                        angle=math.pi / 4.0,
                    )
                    state_cursor += 1
                else:
                    successors = ops.rotate(
                        product,
                        angle=column.quarter_turns * math.pi / 4.0,
                    )
                cursor = 0
                if column.data:
                    for owner in involved_data:
                        output[owner] = successors[cursor]
                        cursor += 1
                if column.ancilla is not None:
                    scratch = successors[cursor]
            if state_cursor != len(states):
                raise AssertionError(
                    "RUS circuit did not consume every T state")
            return (*output, scratch)

        unitary_name = (
            f"{context.lowering.name}_{design.sha256[:12]}_rus_unitary")
        resource_parameters = tuple(
            Parameter(
                f"state{index}",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=resource_annotation,
            ) for index in range(design.t_states_per_attempt))
        unitary_result = tuple[tuple([annotation] * (owner_count + 1))]
        lifted_unitary.__name__ = lifted_unitary.__qualname__ = unitary_name
        lifted_unitary.__module__ = context.lowering.provider.__module__
        lifted_unitary.__signature__ = Signature(
            (
                *block_parameters,
                Parameter(
                    "scratch",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation,
                ),
                *resource_parameters,
            ),
            return_annotation=unitary_result,
        )
        lifted_gadget = gadgets.GadgetDefinition(
            lifted_unitary,
            # The BRS lift acts jointly on the data blocks and architecture-
            # owned scratch.  It is a private action inside an attempt, not a
            # realization of the unconditional user rotation.
            implements=LogicalActionRef(
                "pinnacle_rus_joint_lift",
                owner_count + 1,
            ),
            name=unitary_name,
            type_hints={
                **{
                    parameter.name: annotation for parameter in block_parameters
                },
                "scratch": annotation,
                **{
                    parameter.name: resource_annotation for parameter in resource_parameters
                },
                "return": unitary_result,
            },
            metadata={
                "construction": "certified parsimonious BRS J(V) lift",
                "synthesis_sha256": design.sha256,
                "t_state_rotations": design.t_states_per_attempt,
                "clifford_rotations": design.clifford_rotations_per_attempt,
                "rotation_columns": len(design.columns),
            },
        )

        def attempt(*blocks):
            scratch = ops.allocate_patch(
                context.encoding,
                region=scratch_region,
            )
            scratch = prepare_scratch(scratch)
            states = ops.request_many(
                standard.T_STATE,
                count=design.t_states_per_attempt,
            )
            rotated = lifted_gadget(*blocks, scratch, *states)
            output, scratch = list(rotated[:-1]), rotated[-1]
            scratch, success = scratch_readout(
                scratch,
                analysis=attempt_analysis,
            )
            ops.discard(scratch, reason="Pinnacle RUS attempt complete")

            # The failure Kraus operator is the requested Pauli.  Applying it
            # once restores failure; applying it a second time only on success
            # leaves the successful rotation unchanged.  This uses the native
            # encoded Pauli gadget and avoids a Boolean negation convention.
            corrected = correction(*output)
            if owner_count == 1:
                corrected = (corrected,)
            corrected = ops.cond(
                success,
                then=lambda *values: correction(*values),
                else_=lambda *values: values,
                carries=corrected,
            )
            return (*corrected, success)

        attempt.__name__ = (
            f"{context.lowering.name}_{design.sha256[:12]}_rus_attempt")
        attempt.__qualname__ = attempt.__name__
        attempt.__module__ = context.lowering.provider.__module__
        attempt.__signature__ = Signature(
            block_parameters,
            return_annotation=attempt_result,
        )
        attempt_hints = {
            **{
                parameter.name: annotation for parameter in block_parameters
            },
            "return": attempt_result,
        }
        attempt_protocol = protocols.ProtocolDefinition(
            attempt,
            implements=attempt_objective,
            name=attempt.__name__,
            type_hints=attempt_hints,
            metadata={
                "compiler":
                    "cudaq.logical.architectures.pinnacle_rus",
                "construction":
                    "Bocharov-Roetteler-Svore J(V) attempt",
                "success_effect":
                    "requested_pauli_rotation",
                "failure_effect":
                    "identity_after_pauli_correction",
                "objective_result":
                    "success",
                "success_probability":
                    design.success_probability,
                "success_probability_evidence":
                    f"synthesis:sha256:{design.sha256}",
                "t_states_per_attempt":
                    design.t_states_per_attempt,
                "clifford_rotations_per_attempt":
                    (design.clifford_rotations_per_attempt),
                "logical_measurements_per_attempt":
                    1,
                "readout_model":
                    readout_model.value,
                "scratch_region":
                    scratch_region.name,
                "synthesis_sha256":
                    design.sha256,
            },
        )

        result_annotation = tuple[tuple([annotation] * owner_count)]

        def generated(*blocks):
            values = attempt_protocol(*blocks)
            attempted, success = values[:-1], values[-1]
            return ops.retry(
                attempted,
                until=success,
                max_attempts=max_attempts,
                exhaustion=gadgets.RetryExhaustion.ABORT,
                commit_point=gadgets.before_output(),
            )

        generated.__name__ = (
            f"{context.lowering.name}_{design.sha256[:12]}_rus")
        generated.__qualname__ = generated.__name__
        generated.__module__ = context.lowering.provider.__module__
        generated.__signature__ = Signature(
            block_parameters,
            return_annotation=result_annotation,
        )
        protocol = protocols.ProtocolDefinition(
            generated,
            implements=standard.pauli_rotation,
            name=generated.__name__,
            type_hints={
                **{
                    parameter.name: annotation for parameter in block_parameters
                },
                "return": result_annotation,
            },
            metadata={
                "compiler": "cudaq.logical.architectures.pinnacle_rus",
                "construction": "bounded repeat-until-success rotation",
                "attempt": attempt_protocol.name,
                "max_attempts": max_attempts,
                "exhaustion": "abort",
                # P3 retains the exact bounded attempt, selected predicate
                # profile, controller decision, and precommit retry boundary.
                # Realtime realization remains target-owned and is not part of
                # implied by this projection contract.
                "runtime_replay": "bounded_p3_retry",
            },
        )
        return qec.GeneratedQECArtifact(
            protocol,
            {
                "synthesis_algorithm": "bocharov-roetteler-svore-rus",
                "synthesis_error_metric": "projective_operator_norm",
                "synthesis_requested_precision": design.requested_precision,
                "synthesis_achieved_error": design.achieved_error,
                "synthesis_success_probability": design.success_probability,
                "synthesis_denominator_exponent": design.denominator_exponent,
                "synthesis_single_qubit_t_count": design.single_qubit_t_count,
                "synthesis_lifted_t_count": design.lifted_t_count,
                "synthesis_clifford_rotation_count":
                    (design.clifford_rotations_per_attempt),
                "resource_kind": standard.T_STATE.name,
                "resource_count_per_attempt": design.t_states_per_attempt,
                "logical_measurements_per_attempt": 1,
                "rus_readout_model": readout_model.value,
                "max_attempts": max_attempts,
                "synthesis_sha256": design.sha256,
                "data_owners": owner_count,
                "auxiliary_region": scratch_region.name,
            },
        )

    compile_site.__name__ = f"compile_{encoding.name}_rus_rotation_strategy"
    compile_site.__qualname__ = compile_site.__name__
    return compile_site


def _make_product_rotation_lowering(
    encoding: codes.Encoding,
    *,
    default_rounds: int,
    readout_model: RUSReadoutModel = RUSReadoutModel.EXPLICIT_WSC,
) -> qec.QECLowering:
    return qec.product_rotation.compiler(
        code=encoding,
        clifford=_make_clifford_rotation_strategy(encoding),
        t_injection=_make_pbc_rotation_strategy(encoding),
        synthesis=_make_rus_rotation_strategy(
            encoding,
            default_rounds=default_rounds,
            readout_model=readout_model,
        ),
        plugin="cudaq.logical.architectures.pinnacle",
        version="1.2.0",
        name=f"{encoding.name}_pinnacle_rpp",
    )


@dataclass(frozen=True, slots=True)
class HighRateCSSDefinitions:
    """Typed definitions for one admitted high-rate CSS encoding."""

    instance: PinnacleGBInstance | None
    code: codes.Code
    encoding: codes.Encoding
    prepare_zero: gadgets.GadgetDefinition
    prepare_one: gadgets.GadgetDefinition
    prepare_plus: gadgets.GadgetDefinition
    prepare_minus: gadgets.GadgetDefinition
    memory_round: gadgets.GadgetDefinition
    transversal_cx: gadgets.GadgetDefinition
    cycle_rounds: int
    h_evidence: _CliffordSupport
    s_evidence: _CliffordSupport
    wsc_lowering: qec.QECLowering
    product_rotation_lowering: qec.QECLowering

    def joint_measurement(
        self,
        product: types.PauliProduct,
        *,
        rounds: int | qec.rounds.RoundPolicy | None = None,
    ) -> _wsc.WSCMeasurementBundle:
        """Construct the explicit P2 WSC artifact for inspection or analysis."""

        return _wsc_measurement(self.encoding, product, rounds=rounds)


def _definition_set(
    value,
    *,
    cycle_rounds: int | None = None,
    rus_readout_model: RUSReadoutModel = RUSReadoutModel.EXPLICIT_WSC,
) -> HighRateCSSDefinitions:
    rus_readout_model = _rus_readout_model(rus_readout_model)
    builder = _builder(value)
    code = builder.code
    rounds = (_default_cycle_rounds(code)
              if cycle_rounds is None else _resolve_rounds(code, cycle_rounds))
    return HighRateCSSDefinitions(
        instance=_pinnacle_instance_for_code(code),
        code=code,
        encoding=builder.encoding,
        prepare_zero=builder.prepare_zero(),
        prepare_one=builder.prepare_one(),
        prepare_plus=builder.prepare_plus(),
        prepare_minus=builder.prepare_minus(),
        memory_round=builder.memory(1),
        transversal_cx=builder.transversal_cx(),
        cycle_rounds=rounds,
        h_evidence=builder.h_evidence(),
        s_evidence=builder.s_evidence(),
        wsc_lowering=_make_wsc_lowering(builder.encoding,
                                        default_rounds=rounds),
        product_rotation_lowering=_make_product_rotation_lowering(
            builder.encoding,
            default_rounds=rounds,
            readout_model=rus_readout_model,
        ),
    )


@lru_cache(maxsize=None)
def _preset_definitions(name: str) -> HighRateCSSDefinitions:
    return _definition_set(_pinnacle_gb_by_name(name))


class PinnacleFamily:
    """Cached selector for published presets and admitted generic CSS codes."""

    __slots__ = ()

    @property
    def instances(self):
        return _PINNACLE_GB_INSTANCES

    def __getitem__(self, value: str | int) -> HighRateCSSDefinitions:
        instance = _pinnacle_gb_instance(value)
        return _preset_definitions(instance.name)

    def for_code(
        self,
        value,
        *,
        cycle_rounds: int | None = None,
    ) -> HighRateCSSDefinitions:
        return _definition_set(value, cycle_rounds=cycle_rounds)

    def __repr__(self) -> str:
        return "pinnacle"


_family = PinnacleFamily()


def _architecture(
    definitions: HighRateCSSDefinitions,
    *,
    rus_readout_model: RUSReadoutModel = RUSReadoutModel.EXPLICIT_WSC,
) -> devices.QECArchitecture:
    rus_readout_model = _rus_readout_model(rus_readout_model)
    instance = definitions.instance
    suffix = (instance.name
              if instance is not None else definitions.encoding.name)
    name = "pinnacle_" + "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in suffix)
    rus_scratch = devices.QECRegion(
        name=f"{name}_rus_scratch",
        encoding=definitions.encoding,
        block_capacity=1,
        packing="reserved",
        role="scratch",
        metadata={
            "owner": "pinnacle_rus",
            "lifetime": "attempt_local",
            "logical_ancillas": 1,
        },
    )
    return devices.QECArchitecture(
        name=name,
        encoding=definitions.encoding,
        link_roots=(
            definitions.prepare_zero,
            definitions.prepare_one,
            definitions.prepare_plus,
            definitions.prepare_minus,
            definitions.memory_round,
            definitions.transversal_cx,
            definitions.wsc_lowering,
            definitions.product_rotation_lowering,
        ),
        packing="dense",
        auxiliary_regions=(rus_scratch,),
        metadata={
            "family": "pinnacle",
            "preset": None if instance is None else instance.name,
            "cycle_rounds": definitions.cycle_rounds,
            "wsc_code_distance": "unknown",
            "wsc_circuit_distance": "unknown",
            "pbc_rotation": "typed_t_state_resource_rotate",
            "arbitrary_rotation": "bocharov_roetteler_svore_rus",
            "rus_max_attempts": 32,
            "rus_scratch": "architecture_owned_attempt_local_gb_block",
            "rus_readout_model": rus_readout_model.value,
        },
    )


def for_code(
    code,
    *,
    cycle_rounds: int | None = None,
    rus_readout_model: RUSReadoutModel = RUSReadoutModel.EXPLICIT_WSC,
) -> devices.QECArchitecture:
    """Build a Pinnacle recipe for an admitted high-rate CSS code.

    ``rus_readout_model`` is an explicit architecture assumption. The paper
    model selects one native logical-product measurement per BRS attempt;
    the default retains the constructive serial WSC reference circuit.
    """

    rus_readout_model = _rus_readout_model(rus_readout_model)
    return _architecture(
        _definition_set(
            code,
            cycle_rounds=cycle_rounds,
            rus_readout_model=rus_readout_model,
        ),
        rus_readout_model=rus_readout_model,
    )


def processing_block(
    architecture: devices.QECArchitecture,) -> PinnacleProcessingBlock:
    """Return the cited physical processing-block row for a preset recipe.

    This is a P3 footprint value separate from the complete-P2
    :class:`~cudaq.logical.devices.QECArchitecture`. Generic admitted codes fail closed
    because the Pinnacle paper publishes no physical processing-block row for
    them.
    """

    if not isinstance(architecture, devices.QECArchitecture):
        raise TypeError("pinnacle.processing_block requires a QECArchitecture")
    instance = _pinnacle_instance_for_code(architecture.encoding.code)
    if (instance is None or architecture.name != f"pinnacle_{instance.name}" or
            architecture.metadata.get("family") != "pinnacle" or
            architecture.metadata.get("preset") != instance.name):
        raise ValueError(
            "processing-block footprints require a canonical published "
            "Pinnacle preset architecture")
    return _processing_block_value(instance)


def __getattr__(name: str):
    if name not in {"gb30", "gb62", "gb126", "gb254", "gb510"}:
        raise AttributeError(name)
    architecture = _architecture(_family[name])
    globals()[name] = architecture
    return architecture


def __dir__():
    return sorted((*globals(), *__all__))
