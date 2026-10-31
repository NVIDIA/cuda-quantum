# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Verify and analyze a noisy surface-code memory experiment.

Run the cells in order or execute this file directly. Stim is needed only
when shots is positive. Example 04 adds explicit data-noise placement,
individual gadget models and a comparison with Stim's generated reference.
"""

# %%
from pathlib import Path

import cudaq.logical as cql
from cudaq.logical.architectures import surface
import cudaq_edem as edem

# %%
# Configure the memory experiment.
distance = 3
rounds = 3  # Includes one extraction in preparation and one in readout; >= 2.
p = 0.001
output = Path("surface-memory")
shots = 0  # Set > 0 to sample with Stim (pip install 'stim>=1.16,<2').
seed = 7

# %%
# Select reusable gadgets and connect their code boundaries.
# A gadget supplies the physical circuit and its intended logical action.
# Its contract identifies detector parities and the syndrome at each boundary.
implementation = surface.interleaved_nz_definitions(distance)
prep = surface.gadget_contract(implementation.prepare_zero)
sec = surface.gadget_contract(implementation.syndrome_extraction)
readout = surface.gadget_contract(implementation.measure_z_with_syndrome)
if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 2:
    raise ValueError("rounds must be an integer of at least two")
contracts = (prep, *((sec,) * (rounds - 2)), readout)

# %%
# Verify each gadget's ideal action and detector contract.
# The same SEC gadget can then be used at every intermediate round.
verification = {
    name: cql.gadgets.verify_gadget(contract)
    for name, contract in (("prep", prep), ("SEC", sec), ("readout", readout))
}
assert all(receipt.verified for receipt in verification.values())
print("Verified prep, SEC, readout: "
      "ideal logical actions and detector contracts")

# %%
# Apply standard gate, reset and measurement noise.
# Clifford gates receive depolarization; resets and measurements can flip.
# Extra data noise between operations is an advanced choice in example 04.
noise = cql.compiler.StandardDepolarizingNoise(
    after_clifford_depolarization=p,
    before_measure_flip_probability=p,
    after_reset_flip_probability=p)

# %%
# Emit the physical circuit and derive its composed error model.
# Both paths use the same contracts and noise. An EDEM describes fault effects
# on detectors, logical outcomes and open code boundaries. This preparation-to-
# readout experiment has closed boundaries, so its EDEM projects to a DEM.
text = cql.compiler.emit_stim(contracts, noise=noise)
composed = cql.analysis.derive_edem_chain(contracts, noise=noise)
projected = edem.project_DEM(composed)

# %%
# Export the circuit and the original error model.
# The packed model retains the mutually exclusive alternatives of each Pauli
# channel. Export needs neither Stim nor conversion to independent errors.
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
(output / "memory.stim").write_text(text)
(output / "memory.edemb").write_bytes(projected.to_bytes())
print(f"d={distance}, rounds={rounds}: "
      f"{sum(projected.detectors)} detectors, "
      f"{sum(projected.observables)} observable")
print(f"Artifacts: {output}")

# %%
# Optionally sample physical detector events with Stim.
# These are raw detector events and uncorrected logical flips; no decoder runs.
if shots:
    import stim

    circuit = stim.Circuit(text)
    detectors, observables = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    print(f"Sampled {shots} shots")
    print(f"Shots with detection events: {detectors.any(axis=1).sum()}/{shots}")
    print("Shots with uncorrected observable flips: "
          f"{observables.any(axis=1).sum()}/{shots}")
