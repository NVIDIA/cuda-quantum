# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Private builders for corrected surface Clifford and WSC realizations.

The Clifford methods operate on live typed surface patches.  Product
measurement uses an exact WSC code deformation over a direct sum of the input
surface codes, but its data remain separate live owners: a P2 protocol
allocates a distinct ancilla patch, the nested gadget measures the actual
modified-base/chi/gamma rows, and only that ancilla owner is reset and
discarded.  Algebraic rank, span, sign, and no-proper-factor claims are exact.
The serial bare-ancilla extractor is deliberately not labeled as a
fault-distance-certified lattice-surgery schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from inspect import Parameter, Signature
from typing import Any

from ..codes import Surface
from cudaq.logical.programs.decorators import objective
from cudaq.logical.ops._impl import (
    cx,
    cz,
    discard,
    extract_syndrome,
    h,
    mpp,
    mz,
    parity,
    reset,
    s,
    sdg,
    tick,
    x,
    z,
)
from cudaq.logical.types.values import logical_qubit
from ..gadgets import prepare_plus, prepare_zero
from cudaq.logical.codes import (
    CSSBlock,
    CSSCode,
    Distance,
)
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
from ..std import cx as cx_objective
from ..std import h as h_objective
from ..std import idle as idle_objective
from ..std import s as s_objective
from ._wsc import (
    JointMeasurementEvidence,
    PauliRow,
    _base_stabilizers,
    _build_wsc,
    _global_preparation_objective,
    _in_span,
    _permute_bits,
)


@dataclass(frozen=True, slots=True)
class SurfaceCliffordEvidence:
    """Exact stabilizer/logical certificate for one physical Clifford layer."""

    checks_preserved_with_sign: bool
    logical_x_image: str
    logical_z_image: str
    canonical_action: bool
    construction: str

    def metadata(self):
        return {
            "checks_preserved_with_sign": self.checks_preserved_with_sign,
            "logical_x_image": self.logical_x_image,
            "logical_z_image": self.logical_z_image,
            "canonical_action": self.canonical_action,
            "construction": self.construction,
        }


@dataclass(frozen=True, slots=True)
class SurfaceMeasurementPlan:
    """A distributed live-patch WSC measurement and its evidence."""

    code: Any
    terms: tuple[str, ...]
    requested: PauliRow
    base_checks: tuple[PauliRow, ...]
    merged_checks: tuple[PauliRow, ...]
    chi_checks: tuple[PauliRow, ...]
    gamma_checks: tuple[PauliRow, ...]
    kappa_count: int
    star_kappa_count: int
    rounds: int
    evidence: JointMeasurementEvidence
    auxiliary_code: CSSCode
    gadget: GadgetDefinition


def _mask(row) -> int:
    return sum(1 << qubit for qubit in row)


def _surface_fold_image(row: int, phases: tuple[int, ...],
                        pairs: tuple[tuple[int, int], ...]) -> PauliRow:
    phase_sites = frozenset(phases)
    neighbors: dict[int, list[int]] = {}
    for left, right in pairs:
        neighbors.setdefault(left, []).append(right)
        neighbors.setdefault(right, []).append(left)
    image = PauliRow(0, 0)
    while row:
        bit = row & -row
        qubit = bit.bit_length() - 1
        row ^= bit
        z_support = (1 << qubit) if qubit in phase_sites else 0
        for neighbor in neighbors.get(qubit, ()):
            z_support ^= 1 << neighbor
        image = image * PauliRow(
            1 << qubit,
            z_support,
            int(qubit in phase_sites),
        )
    return image


def _permutation_swaps(
    permutation: tuple[int, ...],) -> tuple[tuple[int, int], ...]:
    """Decompose the source-to-destination permutation into physical SWAPs."""

    visited: set[int] = set()
    swaps = []
    for start in range(len(permutation)):
        if start in visited:
            continue
        cycle = []
        current = start
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = permutation[current]
        swaps.extend((cycle[0], item) for item in cycle[1:])
    return tuple(swaps)


@lru_cache(maxsize=None)
def _surface_direct_sum(distance: int, blocks: int) -> CSSCode:
    base = Surface[distance]
    n = blocks * base.n

    def shifted(rows, block):
        offset = block * base.n
        return tuple(tuple(offset + qubit for qubit in row) for row in rows)

    return CSSCode(
        name=f"surface_{distance}_direct_sum_{blocks}",
        n=n,
        k=blocks,
        d=Distance.unknown(
            "direct-sum algebra is used only to certify a joint deformation"),
        block=CSSBlock(data=n),
        hx=tuple(
            row for block in range(blocks) for row in shifted(base.hx, block)),
        hz=tuple(
            row for block in range(blocks) for row in shifted(base.hz, block)),
        lx=tuple(shifted(base.lx, block)[0] for block in range(blocks)),
        lz=tuple(shifted(base.lz, block)[0] for block in range(blocks)),
        metadata={
            "role": "joint-measurement certificate only",
            "component_code": base.name,
        },
    )


@lru_cache(maxsize=None)
def _surface_auxiliary_code(kappa_count: int) -> CSSCode:
    width = kappa_count + 1
    check = kappa_count
    return CSSCode(
        name=f"surface_wsc_aux_{kappa_count}",
        n=width,
        k=1,
        d=Distance.unknown("temporary WSC ancilla owner"),
        block=CSSBlock(data=width),
        hx=(),
        hz=tuple((qubit,) for qubit in range(kappa_count)),
        lx=((check,),),
        lz=((check,),),
        metadata={
            "role": "temporary WSC kappa and check-ancilla owner",
            "kappa_count": kappa_count,
        },
    )


def _surface_measurement_objective(terms: tuple[str, ...],
                                   name: str,
                                   *,
                                   sign=1):
    names = tuple(f"block{index}" for index in range(len(terms)))

    def intent(*values):
        product = None
        for value, pauli in zip(values, terms):
            factor = {"X": X, "Y": Y, "Z": Z}[pauli](value)
            product = factor if product is None else product @ factor
        return mpp(product if sign > 0 else -product)

    result = tuple[tuple([logical_qubit] * len(terms) + [bool])]
    intent.__name__ = intent.__qualname__ = f"{name}_objective"
    intent.__signature__ = Signature(
        tuple(
            Parameter(
                item, Parameter.POSITIONAL_OR_KEYWORD, annotation=logical_qubit)
            for item in names),
        return_annotation=result,
    )
    intent.__annotations__ = {
        **{
            item: logical_qubit for item in names
        },
        "return": result,
    }
    return objective(intent, name=intent.__name__)


def _distributed_measurement(
    blocks,
    auxiliary,
    row: PauliRow,
    *,
    data_width: int,
    kappa_count: int,
    record: str,
):
    blocks = list(blocks)
    check = kappa_count
    x_only: list[list[int]] = [[] for _ in blocks]
    y_sites: list[list[int]] = [[] for _ in blocks]
    aux_x: list[int] = []
    aux_y: list[int] = []

    for qubit in row.support:
        is_x = bool((row.x >> qubit) & 1)
        is_z = bool((row.z >> qubit) & 1)
        if qubit < len(blocks) * data_width:
            owner, local = divmod(qubit, data_width)
            if is_x and is_z:
                y_sites[owner].append(local)
            elif is_x:
                x_only[owner].append(local)
        else:
            local = qubit - len(blocks) * data_width
            if is_x and is_z:
                aux_y.append(local)
            elif is_x:
                aux_x.append(local)

    for owner in range(len(blocks)):
        if x_only[owner]:
            blocks[owner] = h(blocks[owner].data[tuple(x_only[owner])])
        if y_sites[owner]:
            blocks[owner] = sdg(blocks[owner].data[tuple(y_sites[owner])])
            blocks[owner] = h(blocks[owner].data[tuple(y_sites[owner])])
    if aux_x:
        auxiliary = h(auxiliary.data[tuple(aux_x)])
    if aux_y:
        auxiliary = sdg(auxiliary.data[tuple(aux_y)])
        auxiliary = h(auxiliary.data[tuple(aux_y)])

    auxiliary = reset(auxiliary.data[(check,)])
    if row.hermitian_sign < 0:
        auxiliary = x(auxiliary.data[(check,)])
    for qubit in row.support:
        if qubit < len(blocks) * data_width:
            owner, local = divmod(qubit, data_width)
            blocks[owner], auxiliary = cx(blocks[owner].data[(local,)],
                                          auxiliary.data[(check,)])
        else:
            local = qubit - len(blocks) * data_width
            auxiliary = cx(auxiliary.data[(local,)], auxiliary.data[(check,)])

    for owner in range(len(blocks)):
        if x_only[owner]:
            blocks[owner] = h(blocks[owner].data[tuple(x_only[owner])])
        if y_sites[owner]:
            blocks[owner] = h(blocks[owner].data[tuple(y_sites[owner])])
            blocks[owner] = s(blocks[owner].data[tuple(y_sites[owner])])
    if aux_x:
        auxiliary = h(auxiliary.data[tuple(aux_x)])
    if aux_y:
        auxiliary = h(auxiliary.data[tuple(aux_y)])
        auxiliary = s(auxiliary.data[tuple(aux_y)])
    auxiliary, bit = mz(auxiliary.data[(check,)], record=record)
    return tuple(blocks), auxiliary, bit


def _surface_measurement_artifacts(
    *,
    distance: int,
    terms: tuple[str, ...],
    requested: PauliRow,
    base_checks: tuple[PauliRow, ...],
    merged_checks: tuple[PauliRow, ...],
    chi_checks: tuple[PauliRow, ...],
    kappa_count: int,
    rounds: int,
    evidence: JointMeasurementEvidence,
    sign: int,
):
    base = Surface[distance]
    encoding = base.default_encoding
    auxiliary_code = _surface_auxiliary_code(kappa_count)
    auxiliary_encoding = auxiliary_code.default_encoding
    name = f"surface_{distance}_wsc_{''.join(terms).lower()}"
    intent = _surface_measurement_objective(terms, name, sign=sign)
    block_names = tuple(f"block{index}" for index in range(len(terms)))
    result_annotation = tuple[tuple([patch[encoding]] * len(terms) + [bool])]

    def realization(*values):
        blocks = tuple(values[:-1])
        auxiliary = values[-1]
        auxiliary = reset(auxiliary.data)
        for check_index, row in enumerate(base_checks):
            blocks, auxiliary, _ = _distributed_measurement(
                blocks,
                auxiliary,
                row,
                data_width=base.n,
                kappa_count=kappa_count,
                record=f"pre_base_{check_index}",
            )
        final_chi = None
        chi_offset = len(base_checks)
        for round_index in range(rounds):
            current = []
            for check_index, row in enumerate(merged_checks):
                blocks, auxiliary, bit = _distributed_measurement(
                    blocks,
                    auxiliary,
                    row,
                    data_width=base.n,
                    kappa_count=kappa_count,
                    record=f"merge_{round_index}_{check_index}",
                )
                current.append(bit)
            final_chi = tuple(current[chi_offset:chi_offset + len(chi_checks)])
            tick()
        outcome = parity(*final_chi)

        for round_index in range(rounds):
            current = []
            for check_index, row in enumerate(base_checks):
                blocks, auxiliary, bit = _distributed_measurement(
                    blocks,
                    auxiliary,
                    row,
                    data_width=base.n,
                    kappa_count=kappa_count,
                    record=f"split_{round_index}_{check_index}",
                )
                current.append(bit)
            tick()
        if kappa_count:
            auxiliary, _ = mz(auxiliary.data[tuple(range(kappa_count))],
                              record="split_kappa_z")
        discard((auxiliary,), reason="surface WSC ancilla complete")
        return (*blocks, outcome)

    realization.__name__ = realization.__qualname__ = f"{name}_gadget"
    realization.__signature__ = Signature(
        tuple([
            Parameter(item,
                      Parameter.POSITIONAL_OR_KEYWORD,
                      annotation=patch[encoding]) for item in block_names
        ] + [
            Parameter(
                "auxiliary",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=patch[auxiliary_encoding],
            )
        ]),
        return_annotation=result_annotation,
    )
    realization.__annotations__ = {
        **{
            item: patch[encoding] for item in block_names
        },
        "auxiliary": patch[auxiliary_encoding],
        "return": result_annotation,
    }
    definition = gadget(
        realization,
        implements=intent,
        logical_ports={item: f"{item}.q0" for item in block_names},
        name=realization.__name__,
        metadata={
            "construction": "distributed live-patch WSC surface deformation",
            "merge_rounds": rounds,
            "split_rounds": rounds,
            "kappa_qubits": kappa_count,
            "merged_checks": len(merged_checks),
            "extractor": "serial_bare_ancilla_unverified_fault_distance",
            "code_distance_status": evidence.code_distance.status,
            "circuit_distance_status": evidence.circuit_distance.status,
        },
    )

    return auxiliary_code, definition


class _SurfaceBuilder:
    """Private constructor for one fixed surface-code definition set."""

    def __init__(self, distance: int):
        if not isinstance(distance, int) or isinstance(distance, bool):
            raise TypeError("distance must be an odd Python int at least 3")
        if distance < 3 or distance % 2 == 0:
            raise ValueError("distance must be an odd Python int at least 3")
        self.distance = distance
        self.code = Surface[distance]
        self.encoding = self.code.default_encoding

    @property
    def n(self) -> int:
        return self.code.n

    def prepare_zero(self):
        return prepare_zero(self.encoding,
                            name=f"surface_{self.distance}_prepare_zero")

    def prepare_plus(self):
        return prepare_plus(self.encoding,
                            name=f"surface_{self.distance}_prepare_plus")

    def _flipped_preparation(self, *, plus_basis: bool):
        base = self.prepare_plus() if plus_basis else self.prepare_zero()
        support = self.code.lz[0] if plus_basis else self.code.lx[0]
        operation = z if plus_basis else x
        suffix = "minus" if plus_basis else "one"
        intent = _global_preparation_objective(
            self.code,
            plus if plus_basis else zero,
            operation,
            f"surface_{self.distance}_prepare_{suffix}",
        )
        encoding = self.encoding

        def realization(block):
            block = base(block)
            return operation(block.data[support])

        realization.__name__ = realization.__qualname__ = intent.name
        realization.__annotations__ = {
            "block": patch[encoding],
            "return": patch[encoding],
        }
        return gadget(
            realization,
            implements=intent,
            name=realization.__name__,
            metadata={"logical_eigenvalue": -1},
        )

    def prepare_one(self):
        return self._flipped_preparation(plus_basis=False)

    def prepare_minus(self):
        return self._flipped_preparation(plus_basis=True)

    def memory(self, rounds: int | None = None):
        rounds = self.distance if rounds is None else rounds
        if not isinstance(rounds, int) or isinstance(rounds,
                                                     bool) or rounds <= 0:
            raise ValueError("rounds must be a positive int")
        encoding = self.encoding

        def realization(block):
            for round_index in range(rounds):
                block, _ = extract_syndrome(
                    block, record=f"surface_memory_{round_index}")
            return block

        realization.__name__ = realization.__qualname__ = (
            f"surface_{self.distance}_memory_r{rounds}")
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

    def _h_permutation(self) -> tuple[int, ...]:
        d = self.distance
        # The Mark III surface convention puts rough/smooth boundaries in the
        # orientation exchanged by a clockwise quarter turn, not by the bare
        # transpose used in Adam's original script.
        return tuple(
            (qubit % d) * d + (d - 1 - qubit // d) for qubit in range(d * d))

    def _certify_h(self) -> SurfaceCliffordEvidence:
        permutation = self._h_permutation()
        width = self.code.n
        base = _base_stabilizers(self.code)
        base_span = tuple(row.packed(width) for row in base)
        images = tuple(
            PauliRow(
                _permute_bits(row.z, permutation),
                _permute_bits(row.x, permutation),
                row.phase,
            ) for row in base)
        checks = all(row.phase == 0 and _in_span(row.packed(width), base_span)
                     for row in images)
        lx = _mask(self.code.lx[0])
        lz = _mask(self.code.lz[0])
        x_difference = PauliRow(0, _permute_bits(lx, permutation)) * PauliRow(
            0, lz)
        z_difference = PauliRow(_permute_bits(lz, permutation), 0) * PauliRow(
            lx, 0)
        logical = all(row.phase == 0 and _in_span(row.packed(width), base_span)
                      for row in (x_difference, z_difference))
        return SurfaceCliffordEvidence(
            checks,
            "Z",
            "X",
            checks and logical,
            "transversal H plus boundary-exchanging quarter-turn SWAP",
        )

    def logical_h(self):
        evidence = self._certify_h()
        if not evidence.canonical_action:
            raise ValueError("surface H exact action certificate failed")
        d = self.distance
        swaps = _permutation_swaps(self._h_permutation())
        encoding = self.encoding

        def realization(block):
            block = h(block.data)
            for left, right in swaps:
                for control, target in ((left, right), (right, left), (left,
                                                                       right)):
                    block = cx(block.data,
                               block.data,
                               pairs=((control, target),))
            return block

        realization.__name__ = realization.__qualname__ = f"surface_{d}_h"
        realization.__annotations__ = {
            "block": patch[encoding],
            "return": patch[encoding],
        }
        return gadget(
            realization,
            implements=h_objective,
            name=realization.__name__,
            metadata={
                "construction": evidence.construction,
                "swap_pairs": swaps,
                "exact_action_certificate": evidence.metadata(),
            },
        )

    def _fold_layers(
            self) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        d = self.distance
        original_phases = tuple(row * d + row for row in range(d - 1)) + (
            (d - 2) * d + d - 1,)
        original_pairs = tuple((column * d + row, row * d + column + 1)
                               for row in range(d - 1)
                               for column in range(row, d - 1))
        # Reflect Adam's fold layer into the boundary orientation used by the
        # native Mark III Surface code. This is not cosmetic: the unreflected
        # layer fails to preserve the current stabilizer group.
        reflection = tuple(
            (qubit // d) * d + (d - 1 - qubit % d) for qubit in range(d * d))
        phases = tuple(reflection[qubit] for qubit in original_phases)
        pairs = tuple((reflection[left], reflection[right])
                      for left, right in original_pairs)
        return phases, pairs

    def _certify_s(self) -> SurfaceCliffordEvidence:
        phases, pairs = self._fold_layers()
        width = self.code.n
        base = _base_stabilizers(self.code)
        base_span = tuple(row.packed(width) for row in base)
        x_checks = tuple(_mask(row) for row in self.code.hx)
        checks = all(
            image.phase == 0 and _in_span(image.packed(width), base_span)
            for image in (
                _surface_fold_image(row, phases, pairs) for row in x_checks))
        lx = _mask(self.code.lx[0])
        lz = _mask(self.code.lz[0])
        difference = _surface_fold_image(lx, phases, pairs) * PauliRow(
            lx, lz, 1)
        logical = difference.phase == 0 and _in_span(difference.packed(width),
                                                     base_span)
        return SurfaceCliffordEvidence(
            checks,
            "Y",
            "Z",
            checks and logical,
            "2-local fold-S",
        )

    def fold_s(self):
        evidence = self._certify_s()
        if not evidence.canonical_action:
            raise ValueError("surface S exact action certificate failed")
        phases, pairs = self._fold_layers()
        encoding = self.encoding

        def realization(block):
            block = s(block.data[phases])
            if pairs:
                block = cz(block.data, block.data, pairs=pairs)
            return block

        realization.__name__ = realization.__qualname__ = (
            f"surface_{self.distance}_fold_s")
        realization.__annotations__ = {
            "block": patch[encoding],
            "return": patch[encoding],
        }
        return gadget(
            realization,
            implements=s_objective,
            name=realization.__name__,
            metadata={
                "construction": evidence.construction,
                "phase_qubits": phases,
                "cz_pairs": pairs,
                "exact_action_certificate": evidence.metadata(),
            },
        )

    def transversal_cx(self):
        encoding = self.encoding

        def realization(control, target):
            return cx(control.data, target.data)

        realization.__name__ = realization.__qualname__ = (
            f"surface_{self.distance}_transversal_cx")
        realization.__annotations__ = {
            "control": patch[encoding],
            "target": patch[encoding],
            "return": tuple[patch[encoding], patch[encoding]],
        }
        return gadget(
            realization,
            implements=cx_objective,
            name=realization.__name__,
            metadata={
                "construction": "blockwise transversal CX",
                "physical_pairs": self.code.n,
            },
        )

    def build_joint_measurement(
        self,
        terms: tuple[str, ...],
        *,
        rounds: int | None = None,
        sign: int = 1,
    ) -> SurfaceMeasurementPlan:
        if not isinstance(terms, tuple) or not terms:
            raise ValueError(
                "Pauli product must be a nonempty tuple over X, Y, Z")
        if any(term not in ("X", "Y", "Z") for term in terms):
            raise ValueError(
                "Pauli product must be a nonempty tuple over X, Y, Z")
        if sign not in (-1, 1):
            raise ValueError("Pauli-product sign must be +1 or -1")
        rounds = self.distance if rounds is None else rounds
        if not isinstance(rounds, int) or isinstance(rounds,
                                                     bool) or rounds <= 0:
            raise ValueError("rounds must be a positive int")
        direct_sum = _surface_direct_sum(self.distance, len(terms))
        logical_terms = tuple(
            (index, pauli) for index, pauli in enumerate(terms))
        (
            requested,
            base,
            merged,
            chi,
            gamma,
            _selected,
            _incidence,
            kappa_count,
            star_count,
            evidence,
        ) = _build_wsc(direct_sum, logical_terms, rounds, sign=sign)
        auxiliary, definition = _surface_measurement_artifacts(
            distance=self.distance,
            terms=terms,
            requested=requested,
            base_checks=base,
            merged_checks=merged,
            chi_checks=chi,
            kappa_count=kappa_count,
            rounds=rounds,
            evidence=evidence,
            sign=sign,
        )
        return SurfaceMeasurementPlan(
            code=direct_sum,
            terms=terms,
            requested=requested,
            base_checks=base,
            merged_checks=merged,
            chi_checks=chi,
            gamma_checks=gamma,
            kappa_count=kappa_count,
            star_kappa_count=star_count,
            rounds=rounds,
            evidence=evidence,
            auxiliary_code=auxiliary,
            gadget=definition,
        )


__all__ = []
