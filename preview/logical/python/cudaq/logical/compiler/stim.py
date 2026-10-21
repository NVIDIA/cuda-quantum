# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Annotated Stim export for closed, sequential single-block profile chains."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

from .clifford import project_clifford
from .compile import compile


@dataclass(frozen=True, slots=True)
class StandardDepolarizingNoise:
    """Standard circuit-level depolarization with reset and measurement flips.

    Round-start data noise requires explicit instruction sites in
    :func:`emit_stim`. Depolarizing channels retain their categorical Pauli
    alternatives; no independent-Pauli approximation is made.
    """

    after_clifford_depolarization: float = 0.0
    before_round_data_depolarization: float = 0.0
    before_measure_flip_probability: float = 0.0
    after_reset_flip_probability: float = 0.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value,
                                                          (int, float)) or
                    not isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite in [0, 1]")
            object.__setattr__(self, name, float(value))


def emit_stim(profiles,
              *,
              noise: StandardDepolarizingNoise | None = None,
              round_starts: Mapping[int, Sequence[str]] | None = None) -> str:
    """Export a closed single-block chain of ``GadgetProfile`` values.

    The first occurrence prepares one encoded block, interior occurrences
    preserve it, and the last destroys it. Adjacent boundaries must use the
    same code presentation. Output virtual syndromes are substituted into the
    next occurrence's detector rows. Only the last occurrence may declare
    observables; postselection is unsupported. Construction checks structure,
    not ideal detector-contract correctness.

    ``round_starts`` maps occurrence indices to native instruction site IDs
    before which round-start data noise is inserted. These sites are explicit
    because a round may begin inside preparation or readout. Gate, reset and
    measurement noise is attached to physical operations, not encoded calls.
    The source profiles and their compiled realizations remain unchanged.

    Returns annotated circuit text directly, without a simulator dependency.
    """
    from cudaq.logical.gadgets import GadgetProfile

    profiles = tuple(profiles)
    if len(profiles) < 2 or any(
            not isinstance(profile, GadgetProfile) for profile in profiles):
        raise TypeError(
            "emit_stim requires a closed chain of GadgetProfile values")
    if noise is None:
        noise = StandardDepolarizingNoise()
    if not isinstance(noise, StandardDepolarizingNoise):
        raise TypeError("noise must be StandardDepolarizingNoise")
    starts = dict(round_starts or {})
    if any(not isinstance(i, int) or isinstance(i, bool) or
           not 0 <= i < len(profiles) for i in starts):
        raise ValueError("round-start occurrence index is out of range")
    if noise.before_round_data_depolarization and not any(starts.values()):
        raise ValueError(
            "round-start data noise requires explicit round_starts")

    lines = []
    measurement_count = 0

    def append(kind, targets=(), argument=None):
        head = kind if argument is None else f"{kind}({argument})"
        lines.append(" ".join([head, *(str(target) for target in targets)]))

    next_carrier = 0
    previous_port = None
    previous_presentation = None
    virtual = ()
    detectors = []
    observables = []

    for occurrence, profile in enumerate(profiles):
        if profile.success:
            raise ValueError("Stim chain export does not support postselection")
        if profile.observables and occurrence != len(profiles) - 1:
            raise ValueError("only the final profile may declare observables")
        built = compile(profile)
        projection = project_clifford(built, gadget=profile.gadget.name)
        inputs, outputs = projection.interface.inputs, projection.interface.outputs
        expected_inputs = 0 if occurrence == 0 else 1
        expected_outputs = 0 if occurrence == len(profiles) - 1 else 1
        if len(inputs) != expected_inputs or len(outputs) != expected_outputs:
            raise ValueError("profiles must form a closed single-block chain")
        carriers = {}
        if inputs:
            presentation, = profile.input_profiles.values()
            if presentation is not previous_presentation:
                raise ValueError(
                    "adjacent profiles must use the same code presentation")
            current = dict(inputs[0].partitions)
            if set(current) != set(previous_port) or any(
                    len(current[name]) != len(previous_port[name])
                    for name in current):
                raise ValueError(
                    "adjacent physical carrier partitions do not match")
            for name, local in current.items():
                carriers.update(zip(local, previous_port[name]))
        for local in range(projection.interface.carrier_count):
            if local not in carriers:
                carriers[local] = next_carrier
                next_carrier += 1

        selected = tuple(starts.get(occurrence, ()))
        if (isinstance(starts.get(occurrence), str) or
                any(not isinstance(site, str) for site in selected) or
                len(set(selected)) != len(selected)):
            raise ValueError(
                "round starts must be distinct instruction site IDs")
        available = {op.site_id for op in projection.instructions}
        if not set(selected) <= available:
            raise ValueError(
                "round start does not name a projected instruction")
        boundary, = inputs or outputs
        data = [carriers[q] for q in boundary.data_carriers]
        measurement_offset = measurement_count
        for op in projection.instructions:
            if op.site_id in selected and noise.before_round_data_depolarization:
                append("DEPOLARIZE1", data,
                       noise.before_round_data_depolarization)
            targets = [carriers[q] for q in op.carriers]
            if op.kind == "MPP":
                if noise.before_measure_flip_probability:
                    raise ValueError(
                        "measurement-flip noise on MPP requires an explicit fault model"
                    )
                product = "*".join(
                    f"{pauli}{carriers[q]}" for q, pauli in op.paulis)
                append("MPP", [("!" if op.inverted else "") + product])
                measurement_count += 1
            elif op.kind in ("M", "MX"):
                if noise.before_measure_flip_probability:
                    append("X_ERROR" if op.kind == "M" else "Z_ERROR", targets,
                           noise.before_measure_flip_probability)
                append(op.kind, targets)
                measurement_count += len(targets)
            elif op.kind in ("R", "RX"):
                append(op.kind, targets)
                if noise.after_reset_flip_probability:
                    append("X_ERROR" if op.kind == "R" else "Z_ERROR", targets,
                           noise.after_reset_flip_probability)
            else:
                binary = op.kind in ("CX", "CZ", "SWAP")
                width = 2 if binary else 1
                for start in range(0, len(targets), width):
                    group = targets[start:start + width]
                    append(op.kind, group)
                    if noise.after_clifford_depolarization:
                        append("DEPOLARIZE2" if binary else "DEPOLARIZE1",
                               group, noise.after_clifford_depolarization)

        def lower(parity):
            support = set()
            constant = parity.constant
            for record in parity.records:
                index = measurement_offset + projection.record_indices[
                    record.name]
                support.symmetric_difference_update((index,))
            for term in parity.input_syndromes:
                incoming, offset = virtual[term.index]
                support.symmetric_difference_update(incoming)
                constant ^= offset
            return frozenset(support), constant

        detectors.extend(lower(row.parity) for row in profile.detectors)
        observables.extend(
            (row.index, lower(row.parity)) for row in profile.observables)
        if outputs:
            previous_presentation, = profile.output_profiles.values()
            width = previous_presentation.effective_stabilizers.nrows
            rows = {
                row.index: lower(row.parity) for row in profile.output_syndromes
            }
            if set(rows) != set(range(width)):
                raise ValueError("every output virtual syndrome must be bound")
            virtual = tuple(rows[i] for i in range(width))
            previous_port = {
                name: tuple(carriers[q] for q in values)
                for name, values in outputs[0].partitions
            }

    constant_index = None
    if any(row[1] for row in detectors) or any(
            row[1][1] for row in observables):
        constant_index = measurement_count
        append("MPAD", [1])
        measurement_count += 1

    def targets(row):
        support, constant = row
        indices = sorted(support)
        if constant:
            indices.append(constant_index)
        return [f"rec[{i - measurement_count}]" for i in indices]

    for row in detectors:
        append("DETECTOR", targets(row))
    for index, row in observables:
        append("OBSERVABLE_INCLUDE", targets(row), index)
    return "\n".join(lines) + "\n"


__all__ = ["StandardDepolarizingNoise", "emit_stim"]
