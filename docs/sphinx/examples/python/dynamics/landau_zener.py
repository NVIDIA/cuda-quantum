import cudaq
from cudaq import spin, operators, ScalarOperator, Schedule, ScipyZvodeIntegrator
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# This example simulates the so-called Landau–Zener transition: given a time-dependent Hamiltonian such that the energy separation
# of the two states is a linear function of time, an analytical formula exists to calculate
# the probability of finding the system in the excited state after the transition.

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Define some shorthand operators
sx = spin.x(0)
sz = spin.z(0)
sm = operators.annihilate(0)
sm_dag = operators.create(0)

# Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
dimensions = {0: 2}

# Decoherence parameters
gamma_sm = 0.0001  # relaxation rate
gamma_sz = 0.005  # `dephasing`  rate

# Landau–Zener Hamiltonian:
# `[[-alpha*t, g], [g, alpha*t]] = g * pauli_x - alpha * t * pauli_z`
g = 0.5 * 2 * np.pi
alpha = 2.0 * 2 * np.pi

# Analytical formula
lz_formula_p0 = np.exp(-np.pi * g**2 / (alpha))
lz_formula_p1 = 1.0 - lz_formula_p0

# Hamiltonian
hamiltonian = g * sx - alpha * ScalarOperator(lambda t: t) * sz

# collapse operators: relaxation and `dephasing`
c_op_list = [np.sqrt(gamma_sm) * sm, np.sqrt(gamma_sz) * sz]

# Initial state of the system (ground state)
psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

# Schedule of time steps (simulating a long time range)
steps = np.linspace(-20.0, 20.0, 5000)
schedule = Schedule(steps, ["t"])

# Run the simulation.
evolution_result = cudaq.evolve(hamiltonian,
                                dimensions,
                                schedule,
                                psi0,
                                observables=[operators.number(0)],
                                collapse_operators=c_op_list,
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
plt.savefig('landau_zener.png', dpi=fig.dpi)
