# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Annotated Stim export for closed, sequential single-block profile chains."""

from __future__ import annotations

from typing import Mapping, Sequence

from .clifford import project_clifford
from .compile import compile
from .noise import StandardDepolarizingNoise, _standard_noise_placement


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

    Known limitation: ``fabric.permute`` projects as ordinary SWAPs, which
    receive depolarization when after-Clifford noise is enabled despite
    representing a noiseless relabeling. Preserving this distinction through
    projection and noise policies is future work.

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

        before, after = _standard_noise_placement(projection, noise,
                                                  starts.get(occurrence, ()))
        measurement_offset = measurement_count

        def append_noise(channel):
            append(channel.kind, [carriers[q] for q in channel.carriers],
                   channel.probability)

        for index, op in enumerate(projection.instructions):
            for channel in before[index]:
                append_noise(channel)
            targets = [carriers[q] for q in op.carriers]
            if op.kind == "MPP":
                product = "*".join(
                    f"{pauli}{carriers[q]}" for q, pauli in op.paulis)
                append("MPP", [("!" if op.inverted else "") + product])
                measurement_count += 1
            elif op.kind in ("M", "MX"):
                append(op.kind, targets)
                measurement_count += len(targets)
            elif op.kind in ("R", "RX"):
                append(op.kind, targets)
                for channel in after[index]:
                    append_noise(channel)
            else:
                binary = op.kind in ("CX", "CZ", "SWAP")
                width = 2 if binary else 1
                for start in range(0, len(targets), width):
                    group = targets[start:start + width]
                    append(op.kind, group)
                    if after[index]:
                        append_noise(after[index][start // width])

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
