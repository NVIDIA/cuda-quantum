import cudaq
from cudaq import MatrixOperatorElement, operators, boson, Schedule, ScipyZvodeIntegrator
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# This example demonstrates the use of callback functions to define time-dependent operators.

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Consider a simple 2-level system Hamiltonian exhibits the Landau–Zener transition:
# `[[-alpha*t, g], [g, alpha*t]]
# This can be defined as a callback tensor:
g = 2.0 * np.pi
alpha = 10.0 * 2 * np.pi


def callback_tensor(t):
    return np.array([[-alpha * t, g], [g, alpha * t]], dtype=np.complex128)


# Analytical formula
lz_formula_p0 = np.exp(-np.pi * g**2 / (alpha))
lz_formula_p1 = 1.0 - lz_formula_p0

# Let's define the control term as a callback tensor that acts on 2-level systems
operators.define("lz_op", [2], callback_tensor)

# Hamiltonian
hamiltonian = operators.instantiate("lz_op", [0])

# Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
dimensions = {0: 2}

# Initial state of the system (ground state)
psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

# Schedule of time steps (simulating a long time range)
steps = np.linspace(-4.0, 4.0, 10000)
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
ax.plot(steps, lz_formula_p1 * np.ones(np.shape(steps)), 'k')
ax.plot(steps, lz_formula_p0 * np.ones(np.shape(steps)), 'm')
ax.set_xlabel("Time")
ax.set_ylabel("Occupation probability")
ax.set_title("Landau-Zener transition")
ax.legend(("Excited state", "Ground state", "LZ formula (Excited state)",
           "LZ formula (Ground state)"),
          loc=0)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig('tensor_callback.png', dpi=fig.dpi)
