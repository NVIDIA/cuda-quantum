# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Derive and compose gadget EDEMs, then compare with circuit-derived DEMs.

Install Stim for the comparison: python -m pip install 'stim>=1.16,<2'
"""

# %%
# Import native EDEM operations, surface definitions, and lattice geometry.
from math import isclose
from pathlib import Path

import cudaq.logical as cql
from cudaq.logical.architectures import surface
from cudaq.logical.architecture.geometry import rotated_surface_lattice
import cudaq_edem as edem
import stim


# %%
# Describe preparation, memory-round, and readout measurement results.
def detector_profile(gadget, kind, *, code):
    graph = cql.gadgets.ProfileGraph(gadget.interface)
    if kind == "readout":
        # The gadget's declared result supplies the logical observable.
        return graph.freeze()

    records = {
        (family, i): gadget.record(f"{family}{i}.outcome")
        for family, checks in (("x_check", code.hx), ("z_check", code.hz))
        for i in range(len(checks))
    }

    if kind == "prep":
        for i in range(len(code.hz)):
            graph.detector(records["z_check", i])
    else:
        incoming = gadget.interface.inputs.only().syndrome
        for index, record in enumerate(records.values()):
            graph.detector(incoming[index] ^ record)
    graph.bind(gadget.interface.outputs.only().syndrome,
               tuple(records.values()))
    return graph.freeze()


# %%
# Put independent data depolarization immediately before each stabilizer round.
# MPPs are ideal here; no ancilla gate noise or measurement-flip noise is inferred.
def data_noise_placements(profiles, probability):
    placements = {}
    for index, profile in enumerate(profiles):
        if not profile.gadget.interface.outputs.blocks:
            continue
        projection = cql.compiler.project_clifford(cql.compile(profile),
                                                   gadget=profile.gadget.name)
        port, = projection.interface.inputs or projection.interface.outputs
        entry = next(op for op in projection.instructions if op.kind == "MPP")
        placements[index] = (cql.compiler.Depolarize1Placement(
            entry.site_id, port.data_carriers, probability),)
    return placements


# %%
# Format independent DEM mechanisms for Stim without changing probabilities.
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


# %%
# Read coalesced error terms for a support-by-support comparison.
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


# %%
# Configure the memory experiment.
distance = 3
rounds = 3
p = 0.001
# Only explicit data noise is enabled; the native MPP operations are ideal.
noise = cql.compiler.StandardDepolarizingNoise()
data_depolarization = p
# Set an output directory to save exchange artifacts; None leaves them in memory.
output = None

# %%
# Select the built-in surface recipe and its data-qubit geometry.
implementation = surface.definitions(distance)
code = implementation.code
lattice = rotated_surface_lattice(distance)

# %%
# Assemble preparation, repeated stabilizer rounds, and logical readout.
prep = detector_profile(implementation.prepare_zero, "prep", code=code)
sec = detector_profile(implementation.memory_round, "sec", code=code)
readout = detector_profile(implementation.measure_z, "readout", code=code)
# Preparation contributes one round; repeat the memory round for the rest.
profiles = (prep, *((sec,) * (rounds - 1)), readout)
placements = data_noise_placements(profiles, data_depolarization)

# %%
# Emit the circuit with coordinates, detectors, an observable, and data noise.
coordinates = "\n".join(
    f"QUBIT_COORDS({x}, {y}) {q}" for q, (x,
                                          y) in enumerate(lattice.coordinates))
text = coordinates + "\n" + cql.compiler.emit_stim(
    profiles, noise=noise, depolarizations=placements)
assert sum(line.startswith("DETECTOR") for line in text.splitlines()) == (
    len(implementation.code.hz) + (rounds - 1) * (distance**2 - 1))
assert "OBSERVABLE_INCLUDE(0)" in text

# %%
# Obtain the circuit-derived DEM as the reference path.
circuit = stim.Circuit(text)
circuit_dem = circuit.detector_error_model()

# %%
# Derive each distinct gadget EDEM once, using identical operation/data noise.
cache, occurrences = {}, []
for index, profile in enumerate(profiles):
    key = id(profile), placements.get(index, ())
    if key not in cache:
        build = cql.compile(profile)
        projection = cql.compiler.project_clifford(build,
                                                   gadget=profile.gadget.name)
        faults = cql.analysis.standard_depolarizing_faults(
            projection, noise, depolarizations=placements.get(index, ()))
        cache[key] = cql.analysis.derive_edem(build, faults=faults)
    occurrences.append(cache[key])
occurrences = tuple(occurrences)
assert len(cache) == 3

# %%
# Compose occurrence boundaries, then project the closed chain to a DEM.
composed = edem.compose_chain(occurrences)
assert composed.inputs == composed.outputs == []
projected = edem.project_DEM(composed)

# %%
# Factorize categorical laws and coalesce equal complete response columns.
independent = edem.factorize_dem(projected)
reduced, source_columns = edem.coalesce_dem_columns(independent)
assert len(source_columns) == projected.vector_sizes[0]
edem_dem = to_stim_dem(reduced)

# %%
# Validate dimensions, error supports, and probabilities against the circuit DEM.
assert (circuit_dem.num_detectors,
        circuit_dem.num_observables) == (edem_dem.num_detectors,
                                         edem_dem.num_observables)
actual, expected = dem_errors(edem_dem), dem_errors(circuit_dem)
assert actual.keys() == expected.keys()
assert all(
    isclose(actual[k], expected[k], rel_tol=0, abs_tol=1e-12) for k in expected)
print(
    f"DEM agreement at d={distance}, rounds={rounds}: "
    f"{circuit.num_detectors} detectors, {circuit.num_observables} observable")
print(f"{len(occurrences)} occurrences, {len(cache)} derived definitions, "
      f"{sum(f.size for f in composed.faults)} original fault indicators")
print(f"{len(actual)} matching DEM error terms (atol=1e-12)")

# %%
# Optionally save the circuit, DEMs, and packed per-definition/composed models.
if output is not None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "memory.stim").write_text(text)
    (output / "circuit.dem").write_text(str(circuit_dem))
    (output / "composed.dem").write_text(str(edem_dem))
    (output / "composed.edemb").write_bytes(composed.to_bytes())
    (output / "projected.edemb").write_bytes(projected.to_bytes())
    for profile, value in zip(profiles, occurrences):
        (output / f"{profile.gadget.name}.edemb").write_bytes(value.to_bytes())
