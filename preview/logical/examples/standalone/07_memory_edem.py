# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compare circuit-derived and composed gadget-EDEM surface-memory DEMs.

Fault placement uses Logical's shared noise policy. Composition, projection,
exact factorization and column coalescing use native cudaq-edem operations.
The optional Stim adapter only formats their result.
"""

import argparse
from math import isclose
from pathlib import Path
from runpy import run_path

import cudaq.logical as cql
import cudaq_edem as edem


def memory_recipe(distance, rounds):
    factory = run_path(str(
        Path(__file__).with_name("06_surface_memory.py")))["surface_memory"]
    return factory(distance, rounds)


def memory_faults(profile, noise, depolarizations):
    """Use the same operation noise and explicit placements as circuit export."""
    build = cql.compile(profile)
    projection = cql.compiler.project_clifford(build,
                                               gadget=profile.gadget.name)
    faults = cql.analysis.standard_depolarizing_faults(
        projection, noise, depolarizations=depolarizations)
    return build, faults


def derive_memory_edems(memory, noise, *, data_depolarization=0.0):
    """Derive each distinct definition once; repeated SECs reuse its value."""
    cache, occurrences = {}, []
    placements = memory.depolarizations(data_depolarization)
    for index, profile in enumerate(memory.profiles):
        key = id(profile), placements[index]
        if key not in cache:
            build, faults = memory_faults(profile, noise, placements[index])
            cache[key] = cql.analysis.derive_edem(build, faults=faults)
        occurrences.append(cache[key])
    return tuple(occurrences)


def to_stim_dem(model):
    """Format an independent-mechanism DEM without changing its columns or laws.

    Run factorize_dem and coalesce_dem_columns explicitly before formatting.
    This optional adapter only translates row IDs and writes instructions.
    """
    import stim

    if not isinstance(model, edem.DEM):
        raise TypeError("Stim export requires a DEM from edem.project_DEM")
    if any(model.successes):
        raise ValueError("Stim DEM export does not support postselection")
    if any(group.kind != edem.GroupKind.INDEPENDENT
           for fault in model.faults
           for group in fault.groups):
        raise ValueError("run edem.factorize_dem before Stim formatting")
    offsets, targets, start = [], [], 0
    for fault in model.faults:
        offsets.append(start)
        targets.extend([] for _ in range(fault.size))
        start += fault.size
    row_offsets = {}
    for kind, sizes in ((edem.PortKind.DETECTOR, model.detectors),
                        (edem.PortKind.OBSERVABLE, model.observables)):
        start = 0
        for index, size in enumerate(sizes):
            row_offsets[kind, index] = start
            start += size
    for block in model.blocks:
        for row, column in block.matrix.entries:
            index = row_offsets[block.output.kind, block.output.index] + row
            target = (stim.target_relative_detector_id(index)
                      if block.output.kind == edem.PortKind.DETECTOR else
                      stim.target_logical_observable_id(index))
            targets[offsets[block.input.index] + column].append(target)
    result = stim.DetectorErrorModel()
    for offset, fault in zip(offsets, model.faults):
        for group in fault.groups:
            for column, probability in zip(group.fault_ids,
                                           group.probabilities):
                result.append(
                    stim.DemInstruction("error", [probability],
                                        targets[offset + column]))
    for row in range(sum(model.detectors)):
        result.append("detector", [], [stim.target_relative_detector_id(row)])
    for row in range(sum(model.observables)):
        result.append("logical_observable", [],
                      [stim.target_logical_observable_id(row)])
    return result


def dem_errors(model):
    """Read already-coalesced error terms, ignoring annotation instructions."""
    errors = {}
    for instruction in model.flattened():
        if instruction.type != "error":
            continue
        support = tuple(sorted(str(t) for t in instruction.targets_copy()))
        if support in errors:
            raise ValueError("DEM comparison requires coalesced error terms")
        errors[support] = instruction.args_copy()[0]
    return errors


def compare(distance=3, rounds=3, noise=None, *, data_depolarization=.001):
    """Build both paths from identical gadget profiles and noise parameters."""
    import stim

    noise = noise or cql.compiler.StandardDepolarizingNoise(.001, .001, .001)
    memory = memory_recipe(distance, rounds)
    circuit = stim.Circuit(
        memory.to_stim(noise, data_depolarization=data_depolarization))
    circuit_dem = circuit.detector_error_model()
    occurrences = derive_memory_edems(memory,
                                      noise,
                                      data_depolarization=data_depolarization)
    composed = edem.compose_chain(occurrences)
    projected = edem.project_DEM(composed)
    independent = edem.factorize_dem(projected)
    reduced, source_columns = edem.coalesce_dem_columns(independent)
    assert len(source_columns) == projected.vector_sizes[0]
    edem_dem = to_stim_dem(reduced)
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
