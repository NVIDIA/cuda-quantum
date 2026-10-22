# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compare circuit-derived and composed gadget-EDEM surface-memory DEMs.

Composition and DEM projection use the native cudaq-edem library. This example
adds noise placement and an exact, optional Stim DEM adapter.
"""

import argparse
from collections import Counter
from itertools import product
from math import expm1, fsum, isclose, log1p
from pathlib import Path
from runpy import run_path

import cudaq.logical as cql
from cudaq.logical.analysis import ElementaryFault, FaultSet, PauliInsertion
import cudaq_edem as edem


def memory_recipe(distance, rounds):
    factory = run_path(str(
        Path(__file__).with_name("06_surface_memory.py")))["surface_memory"]
    return factory(distance, rounds)


def memory_faults(profile, noise, round_starts):
    """Place the same standard noise channels as the memory circuit exporter.

    Measurement flips use records: this recipe resets check ancillas before
    their next interaction and destructively measures data at its end.
    """
    build = cql.compile(profile)
    projection = cql.compiler.project_clifford(build,
                                               gadget=profile.gadget.name)
    boundary, = projection.interface.inputs or projection.interface.outputs
    faults, groups = [], []

    def channel(after, carriers, probability, depolarizing=True, basis="x"):
        if not probability:
            return
        words = tuple(
            word for word in product("ixyz", repeat=len(carriers))
            if any(p != "i" for p in word)) if depolarizing else ((basis,),)
        start = len(faults)
        for word in words:
            factors = tuple((q, p) for q, p in zip(carriers, word) if p != "i")
            faults.append(
                ElementaryFault(f"fault{len(faults)}", probability / len(words),
                                (PauliInsertion(after, factors),)))
        if len(words) > 1:
            groups.append(tuple(range(start, len(faults))))

    previous = ""
    for op in projection.instructions:
        if op.site_id in round_starts:
            for q in boundary.data_carriers:
                channel(previous, (q,), noise.before_round_data_depolarization)
        if op.kind in ("R", "RX"):
            for q in op.carriers:
                channel(op.site_id, (q,), noise.after_reset_flip_probability,
                        False, "x" if op.kind == "R" else "z")
        elif op.kind in ("M", "MX"):
            if noise.before_measure_flip_probability:
                for lane in range(len(op.carriers)):
                    record = projection.interface.measurements[
                        op.measurement_start + lane].record
                    faults.append(
                        ElementaryFault(f"fault{len(faults)}",
                                        noise.before_measure_flip_probability,
                                        record_flips=(record,)))
        elif op.kind == "MPP":
            raise ValueError(
                "the memory recipe requires single-qubit measurements")
        else:
            width = 2 if op.kind in ("CX", "CZ", "SWAP") else 1
            for start in range(0, len(op.carriers), width):
                channel(op.site_id, op.carriers[start:start + width],
                        noise.after_clifford_depolarization)
        previous = op.site_id
    return build, FaultSet(tuple(faults), tuple(groups))


def derive_memory_edems(memory, noise):
    """Derive each distinct definition once; repeated SECs reuse its value."""
    cache, occurrences = {}, []
    for index, profile in enumerate(memory.profiles):
        key = id(profile), tuple(memory.round_starts[index])
        if key not in cache:
            build, faults = memory_faults(profile, noise,
                                          memory.round_starts[index])
            cache[key] = cql.analysis.derive_edem(build, faults=faults)
        occurrences.append(cache[key])
    return tuple(occurrences)


def to_stim_dem(model):
    """Render a projected DEM with exact independent/depolarizing fault laws.

    Native project_DEM retains the original columns and categorical laws.
    Uniform one/two-qubit Pauli channels admit an exact independent-error
    factorization. Arbitrary exclusive laws and over-mixing depolarization
    are rejected instead of approximated. The EDEM keeps the original laws.
    """
    import stim

    if not isinstance(model, edem.DEM):
        raise TypeError("Stim export requires a DEM from edem.project_DEM")
    if any(model.successes):
        raise ValueError("Stim DEM export does not support postselection")
    detector_count, observable_count = sum(model.detectors), sum(
        model.observables)
    offsets, columns, start = [], [], 0
    for distribution in model.faults:
        offsets.append(start)
        columns.extend([0] * distribution.size)
        start += distribution.size
    row_offsets = {}
    for kind, sizes, start in ((edem.PortKind.DETECTOR, model.detectors, 0),
                               (edem.PortKind.OBSERVABLE, model.observables,
                                detector_count)):
        for index, size in enumerate(sizes):
            row_offsets[kind, index] = start
            start += size
    for block in model.blocks:
        for row, column in block.matrix.entries:
            columns[offsets[block.input.index] + column] ^= 1 << (
                row_offsets[block.output.kind, block.output.index] + row)
    errors = {}

    def error(response, probability):
        if response and probability:
            previous = errors.get(response, 0)
            errors[
                response] = previous + probability - 2 * previous * probability

    for offset, distribution in zip(offsets, model.faults):
        for group in distribution.groups:
            responses = [columns[offset + i] for i in group.fault_ids]
            if group.kind == edem.GroupKind.INDEPENDENT:
                for response, probability in zip(responses,
                                                 group.probabilities):
                    error(response, probability)
                continue
            count = len(responses)
            if count not in (3, 15) or len(set(group.probabilities)) != 1:
                raise ValueError(
                    "exact DEM export requires uniform Pauli channels")
            # A linear image of all Pauli alternatives plus identity is uniform
            # on a GF(2) subspace, including multiplicities when columns collide.
            multiplicities = Counter([0, *responses])
            basis = {}
            for value in multiplicities:
                while value:
                    pivot = value.bit_length()
                    if pivot not in basis:
                        basis[pivot] = value
                        break
                    value ^= basis[pivot]
            if (len(multiplicities) != 1 << len(basis) or
                    len(set(multiplicities.values())) != 1):
                raise ValueError(
                    "exclusive responses are not a full Pauli channel")
            probability = fsum(group.probabilities)
            limit = count / (count + 1)
            if probability > limit:
                raise ValueError(
                    "over-mixing depolarization has no supported exact DEM")
            # Every nontrivial Pauli character meets (count+1)/2 alternatives.
            independent = .5 if probability == limit else -.5 * expm1(
                log1p(-probability / limit) / ((count + 1) / 2))
            for response in responses:
                error(response, independent)

    result = stim.DetectorErrorModel()
    for response, probability in sorted(errors.items()):
        targets = []
        for row in range(detector_count + observable_count):
            if response >> row & 1:
                targets.append(
                    stim.target_relative_detector_id(row) if row <
                    detector_count else stim.
                    target_logical_observable_id(row - detector_count))
        result.append("error", probability, targets)
    for row in range(detector_count):
        result.append("detector", [], [stim.target_relative_detector_id(row)])
    for row in range(observable_count):
        result.append("logical_observable", [],
                      [stim.target_logical_observable_id(row)])
    return result


def dem_errors(model):
    """Combine independent DEM errors with equal detector/observable support."""
    errors = {}
    for instruction in model.flattened():
        if instruction.type != "error":
            continue
        support = tuple(sorted(str(t) for t in instruction.targets_copy()))
        p = instruction.args_copy()[0]
        previous = errors.get(support, 0)
        errors[support] = previous + p - 2 * previous * p
    return errors


def compare(distance=3, rounds=3, noise=None):
    """Build both paths from identical gadget profiles and noise parameters."""
    import stim

    noise = noise or cql.compiler.StandardDepolarizingNoise(
        .001, .001, .001, .001)
    memory = memory_recipe(distance, rounds)
    circuit = stim.Circuit(memory.to_stim(noise))
    circuit_dem = circuit.detector_error_model()
    occurrences = derive_memory_edems(memory, noise)
    composed = edem.compose_chain(occurrences)
    projected = edem.project_DEM(composed)
    edem_dem = to_stim_dem(projected)
    if (circuit_dem.num_detectors,
            circuit_dem.num_observables) != (edem_dem.num_detectors,
                                             edem_dem.num_observables):
        raise AssertionError("DEM dimensions differ")
    actual, expected = dem_errors(edem_dem), dem_errors(circuit_dem)
    if actual.keys() != expected.keys() or any(
            not isclose(actual[k], expected[k], rel_tol=0, abs_tol=1e-12)
            for k in expected):
        raise AssertionError("circuit and composed-EDEM DEM errors differ")
    return (memory, occurrences, composed, projected, circuit, circuit_dem,
            edem_dem)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write circuit, DEMs, and packed EDEM values to this directory")
    args = parser.parse_args(argv)
    (memory, occurrences, composed, projected, circuit, circuit_dem,
     edem_dem) = compare(args.distance, args.rounds)
    print(
        f"DEM agreement at d={args.distance}, rounds={args.rounds}: "
        f"{circuit.num_detectors} detectors, {circuit.num_observables} observable"
    )
    print(f"{len(occurrences)} occurrences, "
          f"{len({id(m) for m in occurrences})} derived definitions, "
          f"{sum(f.size for f in composed.faults)} original fault indicators")
    print(f"{len(dem_errors(edem_dem))} matching DEM error terms (atol=1e-12)")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "memory.stim").write_text(str(circuit))
        (args.output / "circuit.dem").write_text(str(circuit_dem))
        (args.output / "composed.dem").write_text(str(edem_dem))
        (args.output / "composed.edemb").write_bytes(composed.to_bytes())
        (args.output / "projected.edemb").write_bytes(projected.to_bytes())
        for profile, value in zip(memory.profiles, occurrences):
            (args.output / f"{profile.gadget.name}.edemb").write_bytes(
                value.to_bytes())


if __name__ == "__main__":
    main()
