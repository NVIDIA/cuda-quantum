# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Checked gadget-to-EDEM analysis with explicit finite Pauli/record faults."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from typing import TYPE_CHECKING, Mapping

from cudaq_edem import EDEM

if TYPE_CHECKING:
    from cudaq.logical.compiler.build import Build


@dataclass(frozen=True, slots=True)
class PauliInsertion:
    """A physical Pauli inserted after one projected Clifford instruction."""

    after: str
    factors: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.after, str):
            raise TypeError("insertion site must be a stable site ID")
        factors = tuple(self.factors)
        if not factors or any(not isinstance(carrier, int) or isinstance(
                carrier, bool) or carrier < 0 or basis not in ("x", "y", "z")
                              for carrier, basis in factors):
            raise ValueError(
                "insertion needs physical carrier and X/Y/Z factors")
        if len({carrier for carrier, _ in factors}) != len(factors):
            raise ValueError("insertion repeats a carrier")
        object.__setattr__(self, "factors", factors)


@dataclass(frozen=True, slots=True)
class ElementaryFault:
    """One mutually composable binary source, independent unless grouped."""

    id: str
    probability: float
    insertions: tuple[PauliInsertion, ...] = ()
    record_flips: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("fault ID must be nonempty")
        if isinstance(self.probability, bool) or not isinstance(
                self.probability, (float, int)) or not isfinite(
                    self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("fault probability must be finite in [0, 1]")
        insertions = tuple(self.insertions)
        flips = tuple(self.record_flips)
        if any(not isinstance(item, PauliInsertion) for item in insertions):
            raise TypeError("fault insertions must be PauliInsertion values")
        if any(not isinstance(path, str) or not path for path in flips):
            raise ValueError("record flips must name stable records")
        if not insertions and not flips:
            raise ValueError("elementary fault must have an effect")
        object.__setattr__(self, "probability", float(self.probability))
        object.__setattr__(self, "insertions", insertions)
        object.__setattr__(self, "record_flips", flips)


@dataclass(frozen=True, slots=True)
class FaultSet:
    """Finite binary faults, with categorical alternatives kept exclusive.

    Exclusive-group probability totals are checked by the EDEM library during
    derivation, using the same policy as direct EDEM construction and loading.
    """

    faults: tuple[ElementaryFault, ...] = ()
    exclusive_groups: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        faults = tuple(self.faults)
        groups = tuple(tuple(group) for group in self.exclusive_groups)
        if any(not isinstance(fault, ElementaryFault) for fault in faults):
            raise TypeError("faults must be ElementaryFault values")
        if len({fault.id for fault in faults}) != len(faults):
            raise ValueError("fault IDs must be unique")
        claimed: set[int] = set()
        for group in groups:
            if len(group) < 2 or any(
                    not isinstance(i, int) or isinstance(i, bool) or
                    not 0 <= i < len(faults) or i in claimed
                    for i in group) or tuple(sorted(group)) != group or len(
                        set(group)) != len(group):
                raise ValueError(
                    "exclusive groups need ordered, disjoint indices")
            claimed.update(group)
        object.__setattr__(self, "faults", faults)
        object.__setattr__(self, "exclusive_groups", groups)

    def add(self, fault: ElementaryFault) -> FaultSet:
        return FaultSet((*self.faults, fault), self.exclusive_groups)

    def pauli_error(self, after: str, carrier: int, basis: str,
                    probability: float) -> FaultSet:
        """Add an independent X_ERROR, Y_ERROR, or Z_ERROR source."""
        if basis not in ("x", "y", "z"):
            raise ValueError("basis must be x, y, or z")
        return self.add(
            ElementaryFault(f"fault{len(self.faults)}:{basis}{carrier}",
                            probability,
                            (PauliInsertion(after, ((carrier, basis),)),)))

    def record_flip(self, record, probability: float) -> FaultSet:
        """Add an independent flip of one exact realization record."""
        if not isinstance(record, str):
            from cudaq.logical.gadgets.records import RecordRef
            if not isinstance(record, RecordRef):
                raise TypeError("record must be a stable path or RecordRef")
            record = record.name
        return self.add(
            ElementaryFault(f"fault{len(self.faults)}:record",
                            probability,
                            record_flips=(record,)))

    def pauli_channel(self, after: str, alternatives: Mapping[str, float],
                      carriers: tuple[int, ...]) -> FaultSet:
        """Add one categorical Pauli channel (e.g. X/Y/Z or two-qubit Pauli)."""
        carriers = tuple(carriers)
        if not carriers or len(set(carriers)) != len(carriers) or any(
                not isinstance(carrier, int) or isinstance(carrier, bool) or
                carrier < 0 for carrier in carriers):
            raise ValueError(
                "channel carriers must be distinct nonnegative indices")
        added = []
        prefix = f"channel{len(self.faults)}"
        seen_words: set[str] = set()
        for word, probability in alternatives.items():
            if not isinstance(word, str) or len(word) != len(carriers) or any(
                    basis not in "ixyz" for basis in word.lower()):
                raise ValueError(
                    "Pauli word must match carriers and use I/X/Y/Z")
            if word.lower() in seen_words:
                raise ValueError("Pauli channel repeats an alternative")
            seen_words.add(word.lower())
            _check_probability(probability)
            factors = tuple((carrier, basis.lower())
                            for carrier, basis in zip(carriers, word)
                            if basis.lower() != "i")
            if not factors:
                if probability == 0:
                    continue
                raise ValueError(
                    "identity is the implicit no-fault alternative")
            added.append(
                ElementaryFault(f"{prefix}:{word.lower()}", float(probability),
                                (PauliInsertion(after, factors),)))
        start = len(self.faults)
        groups = self.exclusive_groups
        if len(added) > 1:
            groups += (tuple(range(start, start + len(added))),)
        return FaultSet((*self.faults, *added), groups)

    def depolarize1(self, after: str, carrier: int,
                    probability: float) -> FaultSet:
        _check_probability(probability)
        return self.pauli_channel(after,
                                  {basis: probability / 3 for basis in "xyz"},
                                  (carrier,))

    def depolarize2(self, after: str, left: int, right: int,
                    probability: float) -> FaultSet:
        _check_probability(probability)
        return self.pauli_channel(
            after, {
                a + b: probability / 15
                for a in "ixyz" for b in "ixyz" if a + b != "ii"
            }, (left, right))

    def to_json(self) -> str:
        payload = {
            "schema": "cudaq.logical.edem-faults/v1",
            "faults": [{
                "id": fault.id,
                "insertions": [{
                    "after":
                        insertion.after,
                    "factors": [{
                        "carrier": carrier,
                        "pauli": basis
                    } for carrier, basis in insertion.factors],
                } for insertion in fault.insertions],
                "record_flips": list(fault.record_flips),
            } for fault in self.faults],
            "exclusive_groups": [
                list(group) for group in self.exclusive_groups
            ],
            "probabilities": [fault.probability for fault in self.faults],
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _check_probability(value: float) -> None:
    if (isinstance(value, bool) or not isinstance(value, (float, int)) or
            not isfinite(value) or not 0 <= value <= 1):
        raise ValueError("fault probability must be finite in [0, 1]")


def standard_depolarizing_faults(projection, noise, *,
                                 depolarizations=()) -> FaultSet:
    """Place the standard circuit noise policy on a Clifford projection.

    Uses the same channel placement as ``compiler.emit_stim``. Additional
    one-qubit noise layers use explicit ``compiler.Depolarize1Placement``
    values with projection-local instruction sites and carriers. Measurement
    noise inserts a physical anticommuting Pauli before measurement, preserving
    its effect even if the measured carrier is subsequently reused without reset.

    The result keeps exclusive one-/two-qubit Pauli alternatives and independent
    reset/measurement faults. No Stim dependency or probability approximation.
    The current projection represents noiseless ``fabric.permute`` as SWAPs;
    these receive the same gate noise as in ``compiler.emit_stim``.
    """
    from itertools import product
    from cudaq.logical.compiler.noise import _standard_noise_placement

    before, after = _standard_noise_placement(projection, noise,
                                              depolarizations)
    faults, groups = [], []

    def place(channel, site):
        if channel.kind.startswith("DEPOLARIZE"):
            start = len(faults)
            words = tuple(
                word for word in product("ixyz", repeat=len(channel.carriers))
                if any(p != "i" for p in word))
            for word in words:
                factors = tuple(
                    (q, p) for q, p in zip(channel.carriers, word) if p != "i")
                faults.append(
                    ElementaryFault(f"fault{len(faults)}",
                                    channel.probability / len(words),
                                    (PauliInsertion(site, factors),)))
            groups.append(tuple(range(start, len(faults))))
        else:
            basis = "x" if channel.kind == "X_ERROR" else "z"
            for q in channel.carriers:
                faults.append(
                    ElementaryFault(f"fault{len(faults)}", channel.probability,
                                    (PauliInsertion(site, ((q, basis),)),)))

    previous = ""
    for index, op in enumerate(projection.instructions):
        for channel in before[index]:
            place(channel, previous)
        for channel in after[index]:
            place(channel, op.site_id)
        previous = op.site_id
    return FaultSet(tuple(faults), tuple(groups))


def gadget_to_edem(source: Build,
                   *,
                   faults: FaultSet = FaultSet(),
                   profile: str | None = None) -> EDEM:
    """Prove a gadget detector contract and return a ``cudaq_edem.EDEM``.

    The source is an authenticated compiler Build containing a linked
    ``GadgetContract``. Select its root profile by default, or the unique
    profile for other roots. Input and output code presentations must contain
    canonical stabilizer-code algebra. Detectors and output virtual-syndrome
    relations must hold ideally and, with success bits and deterministic
    combinations of observables, span every deterministic outcome relation.
    Random action outcomes remain valid observables.

    Fault insertions follow realization-local sites from
    ``compiler.project_clifford``. An empty site means the input boundary;
    record flips accept exact projected paths and their aliases. Derivation
    reads the Build's private snapshot, without inserting noise or changing
    canonical IR. EDEM owns probability validation and the packed format;
    affine offsets stay in the authored contract. The ``cudaq-edem`` package
    supplies the model value and its native library.
    """
    from cudaq.logical.compiler.build import Build
    from cudaq.mlir._mlir_libs import _qlxRuntime

    if not isinstance(source, Build):
        raise TypeError("gadget_to_edem expects a cudaq.logical.compiler.Build")
    if not isinstance(faults, FaultSet):
        raise TypeError("faults must be a FaultSet")
    if profile is None:
        profile = source.root.symbol if source.root.kind == "gadget_contract" else ""
    elif not isinstance(profile, str) or not profile:
        raise TypeError("profile must be a nonempty gadget-contract symbol")
    packed = _qlxRuntime.derive_edem_owned(source._fresh_module(), profile,
                                           faults.to_json())
    return EDEM.adopt(packed)


__all__ = [
    "PauliInsertion", "ElementaryFault", "FaultSet", "gadget_to_edem",
    "standard_depolarizing_faults"
]
