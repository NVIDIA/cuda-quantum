# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Compare circuit-level surface memory with its composed gadget EDEMs.

Stim is an optional example dependency for the reference comparison and
sampling. Set compare_with_stim=False and shots=0 to export without Stim.
"""

# %%
from math import isclose
from pathlib import Path

import cudaq.logical as cql
from cudaq.logical.architectures import surface
import cudaq_edem as edem

# %%
# Configure the memory experiment.
distance = 3
rounds = 3  # Includes one extraction in preparation and one in readout; >= 2.
after_clifford_depolarization = 0.001
before_measure_flip_probability = 0.001
after_reset_flip_probability = 0.001
data_depolarization = 0.001
output = Path("surface-memory-circuit")
compare_with_stim = True  # Requires pip install 'stim>=1.16,<2'.
shots = 0
seed = 7
sampling_mode = "circuit"  # Or `dem`: sample the exactly factorized DEM.

# %%
# Select concrete gadgets with the interleaved N/Z gate schedule. Examples
# 02 and 03 use the ideal-MPP gadgets from surface.definitions(distance).
implementation = surface.interleaved_nz_definitions(distance)
prep = surface.gadget_contract(implementation.prepare_zero)
sec = surface.gadget_contract(implementation.syndrome_extraction)
readout = surface.gadget_contract(implementation.measure_z_with_syndrome)
# Both endpoints include an SEC. The example supplies the remaining rounds.
if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 2:
    raise ValueError("rounds must be an integer of at least two")
if sampling_mode not in ("circuit", "dem"):
    raise ValueError("sampling_mode must be circuit or dem")
contracts = (prep, *((sec,) * (rounds - 2)), readout)

# %%
# Verify the ideal +Z preparation, identity SEC and destructive Z readout,
# together with each complete detector contract. Fault distance is separate.
verification = {
    name: cql.gadgets.verify_gadget(contract)
    for name, contract in (("prep", prep), ("SEC", sec), ("readout", readout))
}
assert all(receipt.verified for receipt in verification.values())
print("Verified prep, SEC, readout: "
      "ideal logical actions and detector contracts")

# %%
# Compile and project each reusable gadget to choose physical fault sites.
builds = {
    contract.gadget: cql.compile(contract) for contract in (prep, sec, readout)
}
projections = {
    contract.gadget:
        cql.compiler.project_clifford(builds[contract.gadget],
                                      gadget=contract.gadget.name)
    for contract in (prep, sec, readout)
}

# %%
# Standard Depolarizing applies gate, reset and measurement noise. The recipe
# additionally places data depolarization after initialization resets, before
# each extraction's first H layer. Readout includes an extraction: its data
# layer goes before that extraction, not after the final measurements. No
# extraction boundaries are inferred by the compiler or the noise policy.
noise = cql.compiler.StandardDepolarizingNoise(
    after_clifford_depolarization=after_clifford_depolarization,
    before_measure_flip_probability=before_measure_flip_probability,
    after_reset_flip_probability=after_reset_flip_probability)
data_sites = {
    contract.gadget:
        surface.data_depol(contract, projections[contract.gadget],
                           data_depolarization)
    for contract in (prep, sec, readout)
}
placements = {
    index: data_sites[contract.gadget]
    for index, contract in enumerate(contracts)
}

# %%
# Follow both paths from the same contracts and physical noise placements.
coordinates = "\n".join(f"QUBIT_COORDS({x}, {y}) {q}"
                        for q, (x, y) in enumerate(implementation.coordinates))
text = coordinates + "\n" + cql.compiler.emit_stim(
    contracts, noise=noise, depolarizations=placements)
models = {}
for contract in (prep, sec, readout):
    projection = projections[contract.gadget]
    faults = cql.analysis.standard_depolarizing_faults(
        projection, noise, depolarizations=data_sites[contract.gadget])
    models[contract.gadget] = cql.analysis.edem_from_gadget(contract.gadget,
                                                            contract=contract,
                                                            faults=faults)
occurrences = tuple(models[contract.gadget] for contract in contracts)
composed = edem.compose_chain(occurrences)
projected = edem.project_DEM(composed)

# %%
# Save the circuit and exact categorical laws before any noise conversion.
# The SEC definition is reusable by other composers, including window tasks.
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
(output / "memory.stim").write_text(text)
(output / "memory.edemb").write_bytes(projected.to_bytes())
for name, contract in (("prep", prep), ("sec", sec), ("mz", readout)):
    (output / f"{name}.edemb").write_bytes(models[contract.gadget].to_bytes())
print(f"d={distance}, rounds={rounds}: "
      f"{sum(projected.detectors)} detectors, "
      f"{sum(projected.observables)} observable, "
      f"{sum(f.size for f in projected.faults)} original fault indicators")
print(f"Artifacts: {output}")

# %%
# Exact conversion and coalescing belong to cudaq-edem. The optional Stim
# adapter only formats the independent model produced by these operations.
if compare_with_stim or (shots and sampling_mode == "circuit"):
    import stim

    circuit = stim.Circuit(text)
if compare_with_stim or (shots and sampling_mode == "dem"):
    independent = edem.factorize_dem(projected)
    reduced, column_map = edem.coalesce_dem_columns(independent)
    formatted = cql.analysis.to_stim_dem(reduced)
if compare_with_stim:
    circuit_dem = circuit.detector_error_model()
    reference = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=data_depolarization,
        after_clifford_depolarization=noise.after_clifford_depolarization,
        before_measure_flip_probability=noise.before_measure_flip_probability,
        after_reset_flip_probability=noise.after_reset_flip_probability)

    def error_terms(model):
        terms = {}
        for instruction in model.flattened():
            if instruction.type != "error":
                continue
            support = tuple(sorted(str(t) for t in instruction.targets_copy()))
            if support in terms:
                raise ValueError(
                    "reference comparison requires coalesced terms")
            terms[support] = instruction.args_copy()[0]
        return terms

    actual = error_terms(formatted)
    for expected in (error_terms(circuit_dem),
                     error_terms(reference.detector_error_model())):
        if actual.keys() != expected.keys():
            raise AssertionError("circuit and composed-EDEM supports differ")
        for support, probability in expected.items():
            if not isclose(
                    actual[support], probability, rel_tol=0, abs_tol=1e-12):
                raise AssertionError(
                    "circuit and composed-EDEM probabilities differ")
    (output / "reference.stim").write_text(str(reference))
    (output / "circuit.dem").write_text(str(circuit_dem))
    (output / "composed.dem").write_text(str(formatted))
    print(f"{len(actual)} DEM error terms match the circuit and Stim reference "
          "(atol=1e-12)")

# %%
# Sampling reports raw detector events and uncorrected logical flips. It does
# not run a decoder or estimate a logical failure rate after correction.
if shots:
    if sampling_mode == "circuit":
        detectors, observables = circuit.compile_detector_sampler(
            seed=seed).sample(shots, separate_observables=True)
    else:
        detectors, observables, _ = formatted.compile_sampler(
            seed=seed).sample(shots)
    print(f"Sampled {shots} shots from the {sampling_mode} model")
    print(f"Shots with detection events: {detectors.any(axis=1).sum()}/{shots}")
    print("Shots with uncorrected observable flips: "
          f"{observables.any(axis=1).sum()}/{shots}")
