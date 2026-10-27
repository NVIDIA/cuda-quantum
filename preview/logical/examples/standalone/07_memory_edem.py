# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Configure a surface-memory experiment and export its packed EDEM.

Optionally convert the result to a Stim DEM and sample it. No Stim or QEC
installation is needed to produce the file.
"""

# %%
# Import Logical, the recipe, and the exchange model library.
from pathlib import Path

import cudaq.logical as cql
from cudaq.logical.architectures import surface
import cudaq_edem as edem

# %%
# Configure the memory experiment.
distance = 3
rounds = 3  # Includes preparation's first stabilizer round; must be >= 1.
p = 0.001
# One categorical Pauli channel per data qubit at each selected round.
# For Z-biased noise, try {"x": p / 20, "y": p / 20, "z": 9 * p / 10}
# with shots = 0; Stim sampling supports uniform channels only.
data_pauli_probabilities = {"x": p / 3, "y": p / 3, "z": p / 3}
# Round 0 is preparation; omit a round here to leave its data noiseless.
noisy_rounds = tuple(range(rounds))
# None means every data qubit; a tuple such as (0, 1) limits the noisy sites.
noisy_data_qubits = None
output = Path("memory-edem")
# Set shots > 0 to sample with Stim (pip install 'stim>=1.16,<2'); exporting
# the model does not require Stim.
shots = 0
seed = 7

# %%
# Assemble recipe-provided profiles; an explicit profile can replace any default.
implementation = surface.definitions(distance)
prep = surface.gadget_contract(implementation.prepare_zero)
sec = surface.gadget_contract(implementation.memory_round)
readout = surface.gadget_contract(implementation.measure_z)
profiles = (prep, *((sec,) * (rounds - 1)), readout)


# %%
# Customize the physical fault model. Stabilizer MPPs and readout are ideal.
# The factory is called once per occurrence; identical fault sets reuse a model.
def memory_faults(index, profile, projection):
    faults = cql.analysis.FaultSet()
    if index not in noisy_rounds:
        return faults
    # Insert before the first stabilizer check, after reset in preparation.
    entry = next(
        i for i, op in enumerate(projection.instructions) if op.kind == "MPP")
    after = projection.instructions[entry - 1].site_id if entry else ""
    port, = projection.interface.inputs or projection.interface.outputs
    carriers = port.data_carriers
    selected = range(
        len(carriers)) if noisy_data_qubits is None else noisy_data_qubits
    for q in selected:
        faults = faults.pauli_channel(after, data_pauli_probabilities,
                                      (carriers[q],))
    return faults


# %%
# Derive and compose using the configured faults. No circuit simulator is used.
composed = cql.analysis.derive_edem_chain(profiles, fault_factory=memory_faults)
projected = edem.project_DEM(composed)

# %%
# Save one closed, projected experiment with its original categorical fault laws.
# Consumers load this exact file; no recipe reconstruction is needed.
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
artifact = output / "memory.edemb"
artifact.write_bytes(projected.to_bytes())
print(f"Exported d={distance}, rounds={rounds}: "
      f"{sum(projected.detectors)} detectors, "
      f"{sum(projected.observables)} observable, "
      f"{sum(f.size for f in projected.faults)} fault indicators")
print(f"Model: {artifact}")

# %%
# Optionally sample detector events and uncorrected observable flips with Stim.
# Stim samples independent errors, so first factor each Pauli channel exactly;
# only uniform channels, such as the default above, factor exactly.
if shots:
    independent = edem.factorize_dem(projected)
    reduced, _ = edem.coalesce_dem_columns(independent)
    stim_dem = cql.analysis.to_stim_dem(reduced)
    sampler = stim_dem.compile_sampler(seed=seed)
    detectors, observables, _ = sampler.sample(shots)
    print(f"Sampled {shots} shots")
    print(f"Shots with detection events: {detectors.any(axis=1).sum()}/{shots}")
    print("Shots with uncorrected observable flips: "
          f"{observables.any(axis=1).sum()}/{shots}")
