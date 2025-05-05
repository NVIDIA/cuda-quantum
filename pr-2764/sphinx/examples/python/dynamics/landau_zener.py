import cudaq
from cudaq import spin, boson, ScalarOperator, Schedule, ScipyZvodeIntegrator
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# This example simulates the so-called Landau–Zener transition: given a time-dependent Hamiltonian such that the energy separation
# of the two states is a linear function of time, an analytical formula exists to calculate
# the probability of finding the system in the excited state after the transition.

# References:
# - https://en.wikipedia.org/wiki/Landau%E2%80%93Zener_formula
# - `The Landau-Zener formula made simple`, `Eric P Glasbrenner and Wolfgang P Schleich 2023 J. Phys. B: At. Mol. Opt. Phys. 56 104001`
# - QuTiP notebook: https://github.com/qutip/qutip-notebooks/blob/master/examples/landau-zener.ipynb

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Define some shorthand operators
sx = spin.x(0)
sz = spin.z(0)
sm = boson.annihilate(0)
sm_dag = boson.create(0)

# Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
dimensions = {0: 2}

# Landau–Zener Hamiltonian:
# `[[-alpha*t, g], [g, alpha*t]] = g * pauli_x - alpha * t * pauli_z`
g = 2 * np.pi
# Analytical equation:
# `P(0) = exp(-pi * g ^ 2/ alpha)`
# The target ground state probability that we want to achieve
target_p0 = 0.75
# Compute `alpha` parameter:
alpha = (-np.pi * g**2) / np.log(target_p0)

# Hamiltonian
hamiltonian = g * sx - alpha * ScalarOperator(lambda t: t) * sz

# Initial state of the system (ground state)
psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

# Schedule of time steps (simulating a long time range)
steps = np.linspace(-2.0, 2.0, 5000)
schedule = Schedule(steps, ["t"])

# Run the simulation.
evolution_result = cudaq.evolve(hamiltonian,
                                dimensions,
                                schedule,
                                psi0,
                                observables=[boson.number(0)],
                                collapse_operators=[],
                                store_intermediate_results=True,
                                integrator=ScipyZvodeIntegrator())

prob1 = [
    exp_vals[0].expectation()
    for exp_vals in evolution_result.expectation_values()
]

prob0 = [1 - val for val in prob1]
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(steps, prob1, 'b', steps, prob0, 'r')
ax.plot(steps, (1.0 - target_p0) * np.ones(np.shape(steps)), 'k')
ax.plot(steps, target_p0 * np.ones(np.shape(steps)), 'm')
ax.set_xlabel("Time")
ax.set_ylabel("Occupation probability")
ax.set_title("Landau-Zener transition")
ax.legend(("Excited state", "Ground state", "LZ formula (Excited state)",
           "LZ formula (Ground state)"),
          loc=0)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
plt.savefig('landau_zener.png', dpi=fig.dpi)
