# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Rotated-surface QEC architecture recipes.

The reusable definitions operate on live typed patches: four preparations,
memory, orientation-correct H and fold-S layers, transversal CX, and an
selectable WSC product measurement whose generated protocol owns its temporary
auxiliary patch. The WSC algebra is exact but is not advertised as geometric
distance-preserving lattice surgery; its serial check extractor has unknown
circuit fault distance. A separately named private guard retains the
standard Mark III lattice-surgery compiler only for its currently demonstrated
positive two-patch ZZ contract.

Magic-state cultivation reuses QLX's audited 15-quarter-turn Mark III
definitions. No recipe is installed as a process-global winner or default.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from inspect import Parameter, Signature
from itertools import product as cartesian_product
import cudaq.logical.architecture as architecture
import cudaq.logical.codes as codes
import cudaq.logical.devices as devices
import cudaq.logical.gadgets as gadgets
import cudaq.logical.ops as ops
import cudaq.logical.protocols as protocols
import cudaq.logical.qec as qec
import cudaq.logical.std as standard
import cudaq.logical.types as types
from cudaq.logical.programs.decorators import objective as _objective

from . import _wsc

__all__ = ("SurfaceDefinitions", "definitions", "wsc")


@dataclass(frozen=True, slots=True)
class SurfaceCliffordEvidence:
    """Exact stabilizer/logical certificate for one physical Clifford layer."""

    checks_preserved_with_sign: bool
    logical_x_image: str
    logical_z_image: str
    canonical_action: bool
    construction: str


@dataclass(frozen=True, slots=True)
class SurfaceWSCMeasurementPlan:
    """A distributed live-patch WSC measurement and its evidence."""

    code: codes.Code
    terms: tuple[str, ...]
    requested: _wsc._LabeledPauli
    base_checks: tuple[_wsc._LabeledPauli, ...]
    merged_checks: tuple[_wsc._LabeledPauli, ...]
    chi_checks: tuple[_wsc._LabeledPauli, ...]
    gamma_checks: tuple[_wsc._LabeledPauli, ...]
    kappa_count: int
    star_kappa_count: int
    rounds: int
    evidence: _wsc.WSCMeasurementEvidence
    auxiliary_code: codes.CSSCode
    gadget: gadgets.GadgetDefinition


def _mask(row) -> int:
    return sum(1 << qubit for qubit in row)


def _permute_bits(row: int, permutation: tuple[int, ...]) -> int:
    output = 0
    while row:
        bit = row & -row
        qubit = bit.bit_length() - 1
        output |= 1 << permutation[qubit]
        row ^= bit
    return output


def _packed_span_basis(rows: tuple[int, ...]) -> tuple[int, ...]:
    """Return one exact echelon basis for packed GF(2) rows.

    Surface Clifford certificates repeatedly ask whether hundreds of transformed
    stabilizers belong to the same stabilizer span.  Reconstructing a dense
    ``GF2Matrix`` and recomputing its rank for every transformed row makes that
    proof quartic in the code distance.  The packed rows are already exact
    binary values, so derive the echelon basis once and reuse it.
    """

    basis: dict[int, int] = {}
    for value in rows:
        while value:
            pivot = value.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = value
                break
            value ^= basis[pivot]
    return tuple(basis[pivot] for pivot in sorted(basis, reverse=True))


def _in_packed_span(row: int, basis: tuple[int, ...]) -> bool:
    """Test membership in a basis returned by :func:`_packed_span_basis`."""

    for value in basis:
        pivot = value.bit_length() - 1
        if (row >> pivot) & 1:
            row ^= value
    return row == 0


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


def _surface_fold_image(
        row: int, phases: tuple[int, ...],
        pairs: tuple[tuple[int, int], ...]) -> _wsc._LabeledPauli:
    phase_sites = frozenset(phases)
    neighbors: dict[int, list[int]] = {}
    for left, right in pairs:
        neighbors.setdefault(left, []).append(right)
        neighbors.setdefault(right, []).append(left)
    image = _wsc._LabeledPauli.from_masks(0, 0)
    while row:
        bit = row & -row
        qubit = bit.bit_length() - 1
        row ^= bit
        z_support = (1 << qubit) if qubit in phase_sites else 0
        for neighbor in neighbors.get(qubit, ()):
            z_support ^= 1 << neighbor
        image = image * _wsc._LabeledPauli.from_masks(
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


def _swap_cnot_sequence(
    swaps: tuple[tuple[int, int], ...],) -> tuple[tuple[int, int], ...]:
    """Expand an ordered SWAP decomposition into its emitted CNOT sequence."""

    return tuple(edge for left, right in swaps
                 for edge in ((left, right), (right, left), (left, right)))


@lru_cache(maxsize=None)
def _surface_direct_sum(distance: int, blocks: int) -> codes.CSSCode:
    base = codes.Surface[distance]
    n = blocks * base.n

    def shifted(rows, block):
        offset = block * base.n
        return tuple(tuple(offset + qubit for qubit in row) for row in rows)

    return codes.CSSCode(
        name=f"surface_{distance}_direct_sum_{blocks}",
        n=n,
        k=blocks,
        d=codes.Distance.unknown(
            "direct-sum algebra is used only to certify a joint deformation"),
        block=codes.CSSBlock(data=n),
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
def _surface_auxiliary_code(kappa_count: int) -> codes.CSSCode:
    width = kappa_count + 1
    check = kappa_count
    return codes.CSSCode(
        name=f"surface_wsc_aux_{kappa_count}",
        n=width,
        k=1,
        d=codes.Distance.unknown("temporary WSC ancilla owner"),
        block=codes.CSSBlock(data=width),
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
            factor = {"X": types.X, "Y": types.Y, "Z": types.Z}[pauli](value)
            product = factor if product is None else product @ factor
        return ops.mpp(product if sign > 0 else -product)

    result = tuple[tuple([types.logical_qubit] * len(terms) + [bool])]
    intent.__name__ = intent.__qualname__ = f"{name}_objective"
    intent.__signature__ = Signature(
        tuple(
            Parameter(
                item,
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=types.logical_qubit,
            ) for item in names),
        return_annotation=result,
    )
    intent.__annotations__ = {
        **{
            item: types.logical_qubit for item in names
        },
        "return": result,
    }
    return _objective(intent, name=intent.__name__)


def _controlled_surface_pauli(blocks, auxiliary, control: int,
                              row: _wsc._LabeledPauli, *, data_width: int):
    """Apply a combined-data Pauli controlled by one auxiliary carrier."""

    blocks = list(blocks)
    for qubit in row.support:
        owner, local = divmod(qubit, data_width)
        if owner >= len(blocks):
            raise ValueError("surface recovery Pauli escapes the data owners")
        if (row.x >> qubit) & 1:
            auxiliary, blocks[owner] = ops.cx(auxiliary.data[(control,)],
                                              blocks[owner].data[(local,)])
        if (row.z >> qubit) & 1:
            auxiliary, blocks[owner] = ops.cz(auxiliary.data[(control,)],
                                              blocks[owner].data[(local,)])
    return tuple(blocks), auxiliary


def _distributed_measurement(
    blocks,
    auxiliary,
    row: _wsc._LabeledPauli,
    *,
    data_width: int,
    kappa_count: int,
    record: str,
    recovery: _wsc._LabeledPauli | None = None,
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
            blocks[owner] = ops.h(blocks[owner].data[tuple(x_only[owner])])
        if y_sites[owner]:
            blocks[owner] = ops.sdg(blocks[owner].data[tuple(y_sites[owner])])
            blocks[owner] = ops.h(blocks[owner].data[tuple(y_sites[owner])])
    if aux_x:
        auxiliary = ops.h(auxiliary.data[tuple(aux_x)])
    if aux_y:
        auxiliary = ops.sdg(auxiliary.data[tuple(aux_y)])
        auxiliary = ops.h(auxiliary.data[tuple(aux_y)])

    auxiliary = ops.reset(auxiliary.data[(check,)])
    if row.hermitian_sign < 0:
        auxiliary = ops.x(auxiliary.data[(check,)])
    for qubit in row.support:
        if qubit < len(blocks) * data_width:
            owner, local = divmod(qubit, data_width)
            blocks[owner], auxiliary = ops.cx(blocks[owner].data[(local,)],
                                              auxiliary.data[(check,)])
        else:
            local = qubit - len(blocks) * data_width
            auxiliary = ops.cx(auxiliary.data[(local,)],
                               auxiliary.data[(check,)])

    for owner in range(len(blocks)):
        if x_only[owner]:
            blocks[owner] = ops.h(blocks[owner].data[tuple(x_only[owner])])
        if y_sites[owner]:
            blocks[owner] = ops.h(blocks[owner].data[tuple(y_sites[owner])])
            blocks[owner] = ops.s(blocks[owner].data[tuple(y_sites[owner])])
    if aux_x:
        auxiliary = ops.h(auxiliary.data[tuple(aux_x)])
    if aux_y:
        auxiliary = ops.h(auxiliary.data[tuple(aux_y)])
        auxiliary = ops.s(auxiliary.data[tuple(aux_y)])
    if recovery is not None:
        blocks, auxiliary = _controlled_surface_pauli(
            blocks,
            auxiliary,
            check,
            recovery,
            data_width=data_width,
        )
    auxiliary, bit = ops.mz(auxiliary.data[(check,)], record=record)
    return tuple(blocks), auxiliary, bit


def _surface_measurement_artifacts(
    *,
    distance: int,
    terms: tuple[str, ...],
    requested: _wsc._LabeledPauli,
    base_checks: tuple[_wsc._LabeledPauli, ...],
    merged_checks: tuple[_wsc._LabeledPauli, ...],
    chi_checks: tuple[_wsc._LabeledPauli, ...],
    kappa_count: int,
    rounds: int,
    evidence: _wsc.WSCMeasurementEvidence,
    frame_recovery: _wsc.LogicalFrameRecovery,
    sign: int,
):
    base = codes.Surface[distance]
    encoding = base.default_encoding
    auxiliary_code = _surface_auxiliary_code(kappa_count)
    auxiliary_encoding = auxiliary_code.default_encoding
    sign_key = "p" if sign > 0 else "m"
    name = (f"surface_{distance}_wsc_{''.join(terms).lower()}_"
            f"{sign_key}_r{rounds}")
    intent = _surface_measurement_objective(terms, name, sign=sign)
    block_names = tuple(f"block{index}" for index in range(len(terms)))
    result_annotation = tuple[tuple([types.patch[encoding]] * len(terms) +
                                    [bool])]
    recovery_specs = {}
    for owner in range(len(terms)):
        offset = owner * base.n
        for basis_index, basis_row in enumerate(base.stabilizer_basis.rows):
            shifted = _wsc._row_from_symplectic(
                basis_row,
                base.n,
                label=f"block{owner}_basis_{basis_index}",
            ).shifted(offset)
            check_index = next(index for index, row in enumerate(base_checks)
                               if row.x == shifted.x and row.z == shifted.z)
            correction = _wsc._row_from_symplectic(
                base.anti_stabilizers.rows[basis_index],
                base.n,
                label=f"block{owner}_split_recovery_{basis_index}",
            ).shifted(offset)
            recovery_specs[check_index] = (
                correction * frame_recovery.split_corrections[check_index])

    def realization(*values):
        blocks = tuple(values[:-1])
        auxiliary = values[-1]
        auxiliary = ops.reset(auxiliary.data)
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
            ops.tick()
        outcome = ops.parity(*final_chi)

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
                    recovery=(recovery_specs[check_index]
                              if round_index == rounds - 1 else None),
                )
                current.append(bit)
            ops.tick()
        if kappa_count:
            for index, correction in enumerate(
                    frame_recovery.kappa_corrections):
                blocks, auxiliary = _controlled_surface_pauli(
                    blocks,
                    auxiliary,
                    index,
                    correction,
                    data_width=base.n,
                )
            auxiliary, _ = ops.mz(
                auxiliary.data[tuple(range(kappa_count))],
                record="split_kappa_z",
            )
        ops.discard((auxiliary,), reason="surface WSC ancilla complete")
        return (*blocks, outcome)

    realization.__name__ = realization.__qualname__ = f"{name}_gadget"
    realization.__signature__ = Signature(
        tuple([
            Parameter(item,
                      Parameter.POSITIONAL_OR_KEYWORD,
                      annotation=types.patch[encoding]) for item in block_names
        ] + [
            Parameter(
                "auxiliary",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=types.patch[auxiliary_encoding],
            )
        ]),
        return_annotation=result_annotation,
    )
    realization.__annotations__ = {
        **{
            item: types.patch[encoding] for item in block_names
        },
        "auxiliary": types.patch[auxiliary_encoding],
        "return": result_annotation,
    }
    definition = gadgets.gadget(
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
            "split_recovery": (
                "coherent canonical anti-stabilizer plus kappa-derived logical frame"
            ),
            "logical_frame_recovery": _wsc._metadata_tree(frame_recovery),
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
        self.code = codes.Surface[distance]
        self.encoding = self.code.default_encoding

    @property
    def n(self) -> int:
        return self.code.n

    def prepare_zero(self):
        return gadgets.prepare_zero(
            self.encoding, name=f"surface_{self.distance}_prepare_zero")

    def prepare_plus(self):
        return gadgets.prepare_plus(
            self.encoding, name=f"surface_{self.distance}_prepare_plus")

    def _flipped_preparation(self, *, plus_basis: bool):
        base = self.prepare_plus() if plus_basis else self.prepare_zero()
        support = self.code.lz[0] if plus_basis else self.code.lx[0]
        operation = ops.z if plus_basis else ops.x
        suffix = "minus" if plus_basis else "one"
        intent = _global_preparation_objective(
            self.code,
            types.plus if plus_basis else types.zero,
            operation,
            f"surface_{self.distance}_prepare_{suffix}",
        )
        encoding = self.encoding

        def realization(block):
            block = base(block)
            return operation(block.data[support])

        realization.__name__ = realization.__qualname__ = intent.name
        realization.__annotations__ = {
            "block": types.patch[encoding],
            "return": types.patch[encoding],
        }
        return gadgets.gadget(
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
            for _ in range(rounds):
                block, _ = ops.extract_syndrome(block)
            return block

        realization.__name__ = realization.__qualname__ = (
            f"surface_{self.distance}_memory_r{rounds}")
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
        base = _wsc._base_stabilizers(self.code)
        base_span = tuple(row.packed(width) for row in base)
        base_basis = _packed_span_basis(base_span)
        images = tuple(
            _wsc._LabeledPauli.from_masks(
                _permute_bits(row.z, permutation),
                _permute_bits(row.x, permutation),
                row.phase,
            ) for row in base)
        checks = all(
            row.phase == 0 and _in_packed_span(row.packed(width), base_basis)
            for row in images)
        lx = _mask(self.code.lx[0])
        lz = _mask(self.code.lz[0])
        x_difference = _wsc._LabeledPauli.from_masks(
            0, _permute_bits(lx, permutation)) * _wsc._LabeledPauli.from_masks(
                0, lz)
        z_difference = _wsc._LabeledPauli.from_masks(
            _permute_bits(lz, permutation), 0) * _wsc._LabeledPauli.from_masks(
                lx, 0)
        logical = all(
            row.phase == 0 and _in_packed_span(row.packed(width), base_basis)
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
            block = ops.h(block.data)
            for control, target in _swap_cnot_sequence(swaps):
                block = ops.cx(block.data,
                               block.data,
                               pairs=((control, target),))
            return block

        realization.__name__ = realization.__qualname__ = f"surface_{d}_h"
        realization.__annotations__ = {
            "block": types.patch[encoding],
            "return": types.patch[encoding],
        }
        return gadgets.gadget(
            realization,
            implements=standard.h,
            name=realization.__name__,
            metadata={
                "construction": evidence.construction,
                "swap_pairs": swaps,
                "exact_action_certificate": _wsc._metadata_tree(evidence),
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
        base = _wsc._base_stabilizers(self.code)
        base_span = tuple(row.packed(width) for row in base)
        base_basis = _packed_span_basis(base_span)
        x_checks = tuple(_mask(row) for row in self.code.hx)
        checks = all(
            image.phase == 0 and
            _in_packed_span(image.packed(width), base_basis) for image in (
                _surface_fold_image(row, phases, pairs) for row in x_checks))
        lx = _mask(self.code.lx[0])
        lz = _mask(self.code.lz[0])
        difference = _surface_fold_image(
            lx, phases, pairs) * _wsc._LabeledPauli.from_masks(lx, lz, 1)
        logical = difference.phase == 0 and _in_packed_span(
            difference.packed(width), base_basis)
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
            block = ops.s(block.data[phases])
            if pairs:
                block = ops.cz(block.data, block.data, pairs=pairs)
            return block

        realization.__name__ = realization.__qualname__ = (
            f"surface_{self.distance}_fold_s")
        realization.__annotations__ = {
            "block": types.patch[encoding],
            "return": types.patch[encoding],
        }
        return gadgets.gadget(
            realization,
            implements=standard.s,
            name=realization.__name__,
            metadata={
                "construction": evidence.construction,
                "phase_qubits": phases,
                "cz_pairs": pairs,
                "exact_action_certificate": _wsc._metadata_tree(evidence),
            },
        )

    def transversal_cx(self):
        encoding = self.encoding

        def realization(control, target):
            return ops.cx(control.data, target.data)

        realization.__name__ = realization.__qualname__ = (
            f"surface_{self.distance}_transversal_cx")
        realization.__annotations__ = {
            "control": types.patch[encoding],
            "target": types.patch[encoding],
            "return": tuple[types.patch[encoding], types.patch[encoding]],
        }
        return gadgets.gadget(
            realization,
            implements=standard.cx,
            name=realization.__name__,
            metadata={
                "construction": "blockwise transversal CX",
                "physical_pairs": self.code.n,
            },
        )

    def logical_cz(self, logical_h, transversal_cx):
        """Derive encoded CZ from the certified surface H and CX gadgets."""

        encoding = self.encoding

        def realization(left, right):
            right = logical_h(right)
            left, right = transversal_cx(left, right)
            right = logical_h(right)
            return left, right

        realization.__name__ = realization.__qualname__ = (
            f"surface_{self.distance}_logical_cz")
        realization.__annotations__ = {
            "left": types.patch[encoding],
            "right": types.patch[encoding],
            "return": tuple[types.patch[encoding], types.patch[encoding]],
        }
        return gadgets.gadget(
            realization,
            implements=standard.cz,
            name=realization.__name__,
            metadata={
                "construction": "H(target); transversal CX; H(target)",
                "dependencies": (
                    logical_h.name,
                    transversal_cx.name,
                ),
            },
        )

    def build_wsc_measurement(
        self,
        terms: tuple[str, ...],
        *,
        rounds: int | None = None,
        sign: int = 1,
    ) -> SurfaceWSCMeasurementPlan:
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
        ) = _wsc._build_wsc(direct_sum, logical_terms, rounds, sign=sign)
        frame_recovery = _wsc._logical_frame_recovery(
            direct_sum,
            requested,
            _incidence,
            kappa_count,
            len(base),
        )
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
            frame_recovery=frame_recovery,
            sign=sign,
        )
        return SurfaceWSCMeasurementPlan(
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


def _surface_rounds(code: codes.Code,
                    rounds: int | qec.rounds.RoundPolicy | None) -> int:
    if rounds is None:
        return qec.rounds.from_code_distance.resolve(code)
    if isinstance(rounds, qec.rounds.RoundPolicy):
        return rounds.resolve(code)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise TypeError(
            "rounds must be a positive int or qec.rounds.RoundPolicy")
    return rounds


def _surface_terms(product: types.PauliProduct) -> tuple[str, ...]:
    if not isinstance(product, types.PauliProduct):
        raise TypeError(
            "surface joint measurement expects a types.PauliProduct")
    if product.identities:
        raise ValueError(
            "surface joint measurement does not accept identity-covered patches"
        )
    operands = tuple(factor.operand for factor in product.factors)
    if any(not isinstance(operand, int) or isinstance(operand, bool)
           for operand in operands):
        raise TypeError(
            "surface Pauli factors must use consecutive integer patch operands")
    if operands != tuple(range(len(operands))):
        raise ValueError(
            "surface Pauli factors must cover consecutive patches 0..arity-1")
    return tuple(factor.pauli for factor in product.factors)


def _canonical_mpp_parameters(site) -> tuple[int, int, int]:
    expected = {"x_mask", "z_mask", "sign"}
    actual = set(site.parameters)
    if actual != expected:
        raise ValueError(
            "surface MPP lowering requires exactly x_mask, z_mask, and sign; "
            f"missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}")
    values = tuple(
        site.parameters[name] for name in ("x_mask", "z_mask", "sign"))
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in values):
        raise TypeError("MPP masks and sign must be non-bool integers")
    return values


def _positive_zz_lowering(lowering: qec.QECLowering) -> qec.QECLowering:
    """Guard the stock seam primitive's exact demonstrated MPP contract."""

    def compile_site(site, context):
        if site.objective_family != "pauli_product_measurement":
            raise ValueError(
                "surface ZZ lowering accepts only P0 MPP action sites")
        x_mask, z_mask, sign = _canonical_mpp_parameters(site)
        if len(context.placements) != 2:
            raise ValueError(
                "surface ZZ lowering requires exactly two logical patches")
        boundaries = tuple(binding.block or binding.placement
                           for binding in context.placements)
        if len(set(boundaries)) != 2:
            raise ValueError(
                "surface ZZ lowering requires two distinct QEC blocks")
        if (x_mask, z_mask, sign) != (0, 0b11, 1):
            raise ValueError(
                "surface example lowering supports only positive two-patch ZZ")
        return lowering.provider(site, context)

    compile_site.__name__ = f"compile_{lowering.name}_positive_zz"
    compile_site.__qualname__ = compile_site.__name__
    return qec.QECLowering(
        compile_site,
        objective_family=lowering.objective_family,
        objective=standard.mpp,
        codes=lowering.codes,
        requires=lowering.requires,
        dependencies=lowering.dependencies,
        plugin="cudaq.logical.architectures.surface_positive_zz",
        version="1.0.0",
        name=f"{lowering.name}_positive_zz",
        policy_schema=lowering.policy_schema,
        metadata={
            **dict(lowering.metadata),
            "admission":
                "positive two-patch ZZ only",
            "delegate":
                lowering.plugin,
        },
    )


def _surface_site_product(site, context) -> tuple[types.PauliProduct, str]:
    """Derive one WSC product and data region from a placed MPP site."""

    if site.objective_family != "pauli_product_measurement":
        raise ValueError(
            "surface WSC lowering accepts only P0 MPP action sites")
    x_mask, z_mask, sign = _canonical_mpp_parameters(site)
    if sign not in (-1, 1):
        raise ValueError("MPP sign must be +1 or -1")

    placements = tuple(context.placements)
    width = len(placements)
    if width == 0 or site.input_arity != width:
        raise ValueError(
            "surface WSC lowering requires one placement per MPP operand")
    support = x_mask | z_mask
    if support != (1 << width) - 1:
        raise ValueError(
            "surface WSC masks must cover the complete action-site boundary")
    if any(binding.binding_kind != "local" for binding in placements):
        raise ValueError(
            "surface WSC lowering currently requires local patch placements")

    boundaries = tuple(
        binding.block or binding.placement for binding in placements)
    if len(set(boundaries)) != width:
        raise ValueError(
            "surface WSC lowering requires one distinct QEC block per operand")
    spaces = {binding.space for binding in placements}
    if len(spaces) != 1:
        raise ValueError(
            "surface WSC auxiliary allocation requires one common logical region"
        )

    product = None
    for position in range(width):
        x_bit = (x_mask >> position) & 1
        z_bit = (z_mask >> position) & 1
        factory = types.Y if x_bit and z_bit else types.X if x_bit else types.Z
        factor = factory(position)
        product = factor if product is None else product @ factor
    assert product is not None
    return (product if sign > 0 else -product), spaces.pop()


def _wsc_auxiliary_region(
    device,
    data_region: str,
    auxiliary_encoding: codes.Encoding,
) -> devices.QECRegion:
    """Resolve one derived encoding inside the bound WSC architecture."""

    architecture_bindings = tuple(
        binding for binding in device.logical_to_qec
        if binding.logical_region.name == data_region and
        binding.architecture is not None)
    if len(architecture_bindings) != 1:
        raise ValueError(
            "surface WSC lowering requires exactly one architecture binding "
            f"for logical region {data_region!r}")
    architecture_binding = architecture_bindings[0]
    matches = tuple(region for region in architecture_binding.auxiliary_regions
                    if region.encoding is auxiliary_encoding)
    if not matches:
        raise ValueError(
            "surface WSC lowering requires a device QEC region bound to the "
            f"exact derived auxiliary encoding {auxiliary_encoding.name!r}")
    if len(matches) != 1:
        names = tuple(region.name for region in matches)
        raise ValueError(
            "surface WSC lowering requires one unambiguous QEC region for "
            f"derived auxiliary encoding {auxiliary_encoding.name!r}; "
            f"got {names!r}")
    return matches[0]


@lru_cache(maxsize=32)
def _wsc_measurement(
    distance: int,
    product: types.PauliProduct,
    rounds: int,
) -> _wsc.WSCMeasurementBundle:
    code = codes.Surface[distance]
    plan = _SurfaceBuilder(distance).build_wsc_measurement(
        _surface_terms(product),
        rounds=rounds,
        sign=product.sign,
    )
    return _wsc.WSCMeasurementBundle(
        product=product,
        realization=plan.gadget,
        analysis=_wsc.measurement_profile(plan,
                                          name=f"{plan.gadget.name}_analysis"),
        evidence=plan.evidence,
        rounds=rounds,
        data_code=code,
        auxiliary_code=plan.auxiliary_code,
        diagnostics=plan,
    )


def _make_wsc_lowering(
    encoding: codes.Encoding,
    *,
    distance: int,
    default_rounds: int,
) -> qec.QECLowering:
    """Create a surface WSC compiler with architecture-owned scratch."""

    def compile_site(site, context):
        product, data_region = _surface_site_product(site, context)
        architecture_binding = next(
            (binding for binding in context.device.logical_to_qec
             if binding.logical_region.name == data_region and
             binding.architecture is not None),
            None,
        )
        if architecture_binding is None:
            raise ValueError(
                "surface WSC lowering requires an architecture-bound data region"
            )
        maximum_weight = int(
            architecture_binding.architecture.metadata["max_product_weight"])
        if len(product.factors) > maximum_weight:
            raise ValueError(
                f"surface WSC architecture admits product weight at most "
                f"{maximum_weight}; got {len(product.factors)}")
        rounds = context.policy.get("rounds", default_rounds)
        resolved_rounds = _surface_rounds(context.code, rounds)
        measurement = _wsc_measurement(
            distance,
            product,
            resolved_rounds,
        )
        auxiliary_encoding = measurement.auxiliary_code.default_encoding
        scratch_region = _wsc_auxiliary_region(
            context.device,
            data_region,
            auxiliary_encoding,
        )
        auxiliary_preparation = gadgets.prepare_zero(
            auxiliary_encoding,
            name=f"{measurement.realization.name}_auxiliary_prepare",
        )
        annotation = types.patch[context.encoding]
        owner_count = len(context.placements)

        def generated(*blocks):
            auxiliary = ops.allocate_patch(
                auxiliary_encoding,
                region=scratch_region,
            )
            auxiliary = auxiliary_preparation(auxiliary)
            return measurement.realization(
                *blocks,
                auxiliary,
                analysis=measurement.analysis,
            )

        product_key = (f"x{int(site.parameters['x_mask']):x}_"
                       f"z{int(site.parameters['z_mask']):x}_"
                       f"{'m' if product.sign < 0 else 'p'}")
        generated.__name__ = f"{context.lowering.name}_{product_key}"
        generated.__qualname__ = generated.__name__
        generated.__module__ = context.lowering.provider.__module__
        parameters = tuple(
            Parameter(
                f"block{index}",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation,
            ) for index in range(owner_count))
        result_annotation = tuple[tuple([annotation] * owner_count + [bool])]
        generated.__signature__ = Signature(
            parameters,
            return_annotation=result_annotation,
        )
        hints = {
            **{
                parameter.name: annotation for parameter in parameters
            },
            "return": result_annotation,
        }
        protocol = protocols.ProtocolDefinition(
            generated,
            implements=standard.mpp,
            name=generated.__name__,
            type_hints=hints,
            metadata={
                "compiler": "cudaq.logical.architectures.surface_wsc",
                "construction": "Webster-Smith-Cohen merged-code measurement",
                "rounds": resolved_rounds,
                "data_region": data_region,
                "scratch_region": scratch_region.name,
                "scratch_ownership": "protocol_local",
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
                    int(site.parameters["sign"]),
                "rounds":
                    resolved_rounds,
                "data_owners":
                    owner_count,
                "auxiliary_region":
                    scratch_region.name,
                "auxiliary_code":
                    measurement.auxiliary_code.name,
                "auxiliary_encoding":
                    auxiliary_encoding.name,
                "kappa_qubits":
                    measurement.kappa_qubits,
                "merged_checks":
                    measurement.merged_check_count,
                "chi_checks":
                    measurement.chi_check_count,
                "code_distance_status":
                    measurement.evidence.code_distance.status,
                "circuit_distance_status":
                    (measurement.evidence.circuit_distance.status),
            },
        )

    compile_site.__name__ = f"compile_{encoding.name}_wsc_mpp"
    compile_site.__qualname__ = compile_site.__name__
    return qec.QECLowering(
        compile_site,
        objective_family="pauli_product_measurement",
        objective=standard.mpp,
        codes=(encoding,),
        plugin="cudaq.logical.architectures.surface_wsc",
        version="1.0.0",
        name=f"{encoding.name}_wsc_mpp",
        policy_schema={"rounds": "positive int"},
        metadata={
            "construction": "Webster-Smith-Cohen merged-code measurement",
            "ownership": "separate data blocks plus protocol-local auxiliary",
            "distance_status": "unknown",
        },
    )


@dataclass(frozen=True, slots=True)
class SurfaceDefinitions:
    """The complete reusable definition set for one surface-code distance."""

    distance: int
    code: codes.Code
    square_patch_footprint: architecture.PhysicalFootprint
    prepare_zero: gadgets.GadgetDefinition
    prepare_one: gadgets.GadgetDefinition
    prepare_plus: gadgets.GadgetDefinition
    prepare_minus: gadgets.GadgetDefinition
    memory_round: gadgets.GadgetDefinition
    measure_z: gadgets.GadgetDefinition
    measure_x: gadgets.GadgetDefinition
    h: gadgets.GadgetDefinition
    fold_s: gadgets.GadgetDefinition
    transversal_cx: gadgets.GadgetDefinition
    cz: gadgets.GadgetDefinition
    surgery: qec.lattice_surgery.SurgeryPrimitives
    wsc_lowering: qec.QECLowering
    lattice_surgery_zz_lowering: qec.QECLowering

    def wsc(self, *, max_product_weight: int = 2) -> devices.QECArchitecture:
        """Return the selectable QEC architecture for these definitions."""

        return wsc(
            distance=self.distance,
            max_product_weight=max_product_weight,
        )

    def wsc_measurement(
        self,
        product: types.PauliProduct,
        *,
        rounds: int | qec.rounds.RoundPolicy | None = None,
    ) -> _wsc.WSCMeasurementBundle:
        """Build an inspectable WSC measurement for a typed patch product.

        Integer operands label separate input patches.  For example,
        ``types.Z(0) @ types.Z(1)`` requests a two-patch ZZ measurement.  The
        realization also takes and consumes one patch of ``auxiliary_code``.
        """

        # Validate before cache lookup. types.PauliProduct intentionally excludes
        # explicit identity coverage from equality/hash, so validating only in
        # the cached builder would make admission depend on cache history.
        _surface_terms(product)
        return _wsc_measurement(
            self.distance,
            product,
            _surface_rounds(self.code, rounds),
        )


@lru_cache(maxsize=None)
def _definitions(distance: int) -> SurfaceDefinitions:
    builder = _SurfaceBuilder(distance)
    code = builder.code
    surgery = qec.lattice_surgery.surface_primitives(code)
    logical_h = builder.logical_h()
    transversal_cx = builder.transversal_cx()
    return SurfaceDefinitions(
        distance=distance,
        code=code,
        square_patch_footprint=architecture.PhysicalFootprint(
            "qubit",
            2 * (distance + 1)**2,
            "square rotated-surface-code patch; arXiv:1905.09749 section 2.14",
        ),
        prepare_zero=gadgets.prepare_zero(code),
        prepare_one=builder.prepare_one(),
        prepare_plus=gadgets.prepare_plus(code),
        prepare_minus=builder.prepare_minus(),
        memory_round=gadgets.css_memory_round(code),
        measure_z=gadgets.logical_measure(code, basis=architecture.Basis.Z),
        measure_x=gadgets.logical_measure(code, basis=architecture.Basis.X),
        h=logical_h,
        fold_s=builder.fold_s(),
        transversal_cx=transversal_cx,
        cz=builder.logical_cz(logical_h, transversal_cx),
        surgery=surgery,
        wsc_lowering=_make_wsc_lowering(
            code.default_encoding,
            distance=distance,
            default_rounds=_surface_rounds(code, None),
        ),
        lattice_surgery_zz_lowering=_positive_zz_lowering(
            qec.lattice_surgery.mpp_compiler(
                code=code,
                max_weight=2,
                prepare=surgery.prepare,
                merge=surgery.merge,
                measure=surgery.measure,
                split=surgery.split,
                ancilla=qec.lattice_surgery.connected_ancilla,
                y_basis=qec.lattice_surgery.local_basis_change,
                rounds=qec.rounds.from_code_distance,
                name=f"{code.name}_mpp",
            ),),
    )


class SurfaceFamily:
    """Indexable family of cached surface implementation definitions."""

    __slots__ = ()

    def __getitem__(self, distance: int) -> SurfaceDefinitions:
        return _definitions(distance)

    def __repr__(self) -> str:
        return "surface"


_family = SurfaceFamily()


def definitions(distance: int) -> SurfaceDefinitions:
    """Return the reusable definitions and physical footprint for ``distance``."""

    return _family[distance]


def _admitted_auxiliary_regions(
    distance: int,
    max_product_weight: int,
) -> tuple[devices.QECRegion, ...]:
    encodings = {}
    paulis = ("X", "Y", "Z")
    for weight in range(1, max_product_weight + 1):
        direct_sum = _surface_direct_sum(distance, weight)
        for factors in cartesian_product(paulis, repeat=weight):
            terms = tuple(enumerate(factors))
            kappa_count = _wsc._wsc_kappa_count(direct_sum, terms)
            encoding = _surface_auxiliary_code(kappa_count).default_encoding
            encodings[encoding.name] = (
                kappa_count,
                encoding,
            )
    return tuple(
        devices.QECRegion(
            name=f"wsc_aux_{kappa_count}",
            encoding=encoding,
            block_capacity=1,
            packing="dense",
            role="scratch",
            metadata={
                "owner": "surface_wsc",
                "lifetime": "protocol_local",
                "kappa_count": kappa_count,
            },
        ) for kappa_count, encoding in sorted(encodings.values()))


@lru_cache(maxsize=None)
def _wsc_cached(
    distance: int,
    max_product_weight: int,
) -> devices.QECArchitecture:
    definitions = _family[distance]
    rounds = _surface_rounds(definitions.code, None)
    return devices.QECArchitecture(
        name=f"surface_wsc_d{distance}_w{max_product_weight}",
        encoding=definitions.code.default_encoding,
        link_roots=(
            definitions.prepare_zero,
            definitions.prepare_one,
            definitions.prepare_plus,
            definitions.prepare_minus,
            definitions.memory_round,
            definitions.measure_z,
            definitions.measure_x,
            definitions.h,
            definitions.fold_s,
            definitions.transversal_cx,
            definitions.cz,
            definitions.wsc_lowering,
        ),
        packing="dense",
        auxiliary_regions=_admitted_auxiliary_regions(
            distance,
            max_product_weight,
        ),
        metadata={
            "family": "rotated_surface",
            "realization": "webster_smith_cohen",
            "distance": distance,
            "max_product_weight": max_product_weight,
            "rounds": rounds,
            "wsc_code_distance": "unknown",
            "wsc_circuit_distance": "unknown",
        },
    )


def wsc(
    *,
    distance: int = 3,
    max_product_weight: int = 2,
) -> devices.QECArchitecture:
    """Return a rotated-surface WSC architecture and its derived scratch."""

    if not isinstance(distance, int) or isinstance(distance, bool):
        raise TypeError(
            "surface.wsc distance must be an odd Python int at least 3")
    if distance < 3 or distance % 2 == 0:
        raise ValueError(
            "surface.wsc distance must be an odd Python int at least 3")
    if (not isinstance(max_product_weight, int) or
            isinstance(max_product_weight, bool)):
        raise TypeError(
            "surface.wsc max_product_weight must be the Python int 1 or 2")
    if max_product_weight not in (1, 2):
        raise ValueError(
            "surface.wsc max_product_weight currently supports exactly 1 or 2")
    return _wsc_cached(distance, max_product_weight)
