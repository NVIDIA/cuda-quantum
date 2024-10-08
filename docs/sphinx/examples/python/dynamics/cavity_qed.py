import cudaq
from cudaq.operator import *
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# Set the target to our dynamics simulator
cudaq.set_target("nvidia-dynamics")

# This example demonstrate a simulation of cavity quantum electrodynamics (interaction between light confined in a reflective cavity and atoms)

# System dimensions: atom (2-level system) and cavity (10-level system)
dimensions = {0: 2, 1: 10}

# Alias for commonly used operators
# Cavity operators
a = operators.annihilate(1)
a_dag = operators.create(1)

# Atom operators
sm = operators.annihilate(0)
sm_dag = operators.create(0)

# Defining the Hamiltonian for the system: self-energy terms and cavity-atom interaction term.
hamiltonian = 2 * np.pi * operators.number(1) + 2 * np.pi * operators.number(
    0) + 2 * np.pi * 0.25 * (sm * a_dag + sm_dag * a)

# Initial state of the system
# Atom in ground state
qubit_state = cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128)

# Cavity in a state which has 5 photons initially
cavity_state = cp.zeros((10, 10), dtype=cp.complex128)
cavity_state[5][5] = 1.0
rho0 = cudaq.State.from_data(cp.kron(qubit_state, cavity_state))

steps = np.linspace(0, 10, 201)
schedule = Schedule(steps, ["time"])

# First, evolve the system without any collapse operators (ideal).
evolution_result = evolve(
    hamiltonian,
    dimensions,
    schedule,
    rho0,
    observables=[operators.number(1), operators.number(0)],
    collapse_operators=[],
    store_intermediate_results=True,
    integrator=ScipyZvodeIntegrator())

# Then, evolve the system with a collapse operator modeling cavity decay (leaking photons)
evolution_result_decay = evolve(
    hamiltonian,
    dimensions,
    schedule,
    rho0,
    observables=[operators.number(1), operators.number(0)],
    collapse_operators=[np.sqrt(0.1) * a],
    store_intermediate_results=True,
    integrator=ScipyZvodeIntegrator())

get_result = lambda idx, res: [
    exp_vals[idx].expectation() for exp_vals in res.expectation_values()
]
ideal_results = [
    get_result(0, evolution_result),
    get_result(1, evolution_result)
]
decay_results = [
    get_result(0, evolution_result_decay),
    get_result(1, evolution_result_decay)
]

fig = plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, ideal_results[0])
plt.plot(steps, ideal_results[1])
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.legend(("Cavity Photon Number", "Atom Excitation Probability"))
plt.title("No decay")

plt.subplot(1, 2, 2)
plt.plot(steps, decay_results[0])
plt.plot(steps, decay_results[1])
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.legend(("Cavity Photon Number", "Atom Excitation Probability"))
plt.title("With decay")

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig('cavity_qed.png', dpi=fig.dpi)
