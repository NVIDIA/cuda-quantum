# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Export and optionally sample a profiled rotated-surface Z-memory circuit."""

# %%
# Import the compiler and surface definitions.
import sys

import cudaq.logical as cql
from cudaq.logical.architectures import surface

# %%
# Configure the memory experiment.
distance = 3
rounds = 3
p = 0.001
# Only explicit data noise is enabled; the native MPP operations are ideal.
noise = cql.compiler.StandardDepolarizingNoise()
data_depolarization = p
# Set shots > 0 to sample with Stim (pip install 'stim>=1.16,<2'); exporting
# text does not require Stim.
shots = 0
seed = 7

# %%
# Select the built-in ideal-MPP surface recipe.
implementation = surface.definitions(distance)
code = implementation.code

# %%
# Assemble preparation, repeated stabilizer rounds, and logical readout.
# For custom profiles, see ../../docs/source/reference/detector-contracts.md
# under "Declare a contract".
prep = surface.gadget_contract(implementation.prepare_zero)
sec = surface.gadget_contract(implementation.memory_round)
readout = surface.gadget_contract(implementation.measure_z)
# Preparation contributes one round; this ideal-MPP readout contributes none.
if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
    raise ValueError("rounds must be an integer of at least one")
profiles = (prep, *((sec,) * (rounds - 1)), readout)

# %%
# The example chooses the noisy occurrences. The per-gadget helper resolves
# the declared before_syndrome site, after preparation resets. Each data qubit
# receives a categorical X/Y/Z channel of total probability data_depolarization.
projections = {
    contract.gadget:
        cql.compiler.project_clifford(cql.compile(contract),
                                      gadget=contract.gadget.name)
    for contract in (prep, sec)
}
data_sites = {
    contract.gadget:
        surface.data_depol(contract, projections[contract.gadget],
                           data_depolarization) for contract in (prep, sec)
}
placements = {
    index: data_sites[contract.gadget]
    for index, contract in enumerate(profiles[:-1])
}

# %%
# Emit the circuit with coordinates, detectors, an observable, and data noise.
coordinates = "\n".join(f"QUBIT_COORDS({x}, {y}) {q}"
                        for q, (x, y) in enumerate(implementation.coordinates))
text = coordinates + "\n" + cql.compiler.emit_stim(
    profiles, noise=noise, depolarizations=placements)
assert sum(line.startswith("DETECTOR") for line in text.splitlines()) == (
    len(implementation.code.hz) + (rounds - 1) * (distance**2 - 1))
assert "OBSERVABLE_INCLUDE(0)" in text
print(text, end="")

# %%
# Optionally sample detector events and uncorrected observable flips with Stim.
# Summaries go to stderr so stdout stays valid circuit text for a .stim file.
if shots:
    import stim

    circuit = stim.Circuit(text)
    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(shots, separate_observables=True)
    print(f"Sampled {shots} shots", file=sys.stderr)
    print(f"Shots with detection events: {detectors.any(axis=1).sum()}/{shots}",
          file=sys.stderr)
    print(
        "Shots with uncorrected observable flips: "
        f"{observables.any(axis=1).sum()}/{shots}",
        file=sys.stderr)
