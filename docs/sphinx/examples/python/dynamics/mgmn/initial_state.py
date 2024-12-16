import cudaq
from cudaq import operators, Schedule, RungeKuttaIntegrator
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# This example demonstrate a simulation of cavity quantum electrodynamics (interaction between light confined in a reflective cavity and atoms)

# System dimensions: atom (2-level system) and cavity (1024-level system)
cavity_dim = 1024
dimensions = {0: 2, 1: cavity_dim}

# Alias for commonly used operators
# Cavity operators
a = operators.annihilate(1)
a_dag = operators.create(1)

# Atom operators
sm = operators.annihilate(0)
sm_dag = operators.create(0)

# Defining the Hamiltonian for the system: self-energy terms and cavity-atom interaction term.
# This is the so-called Jaynes-Cummings model:
# https://en.wikipedia.org/wiki/Jaynes%E2%80%93Cummings_model
hamiltonian = 2 * np.pi * operators.number(1) + 2 * np.pi * operators.number(
    0) + 2 * np.pi * 0.25 * (sm * a_dag + sm_dag * a)

rho0 = cudaq.operator.InitialState.ZERO

steps = np.linspace(0, 10, 201)
schedule = Schedule(steps, ["time"])

evolution_result_decay = cudaq.evolve(
    hamiltonian,
    dimensions,
    schedule,
    rho0,
    observables=[operators.number(1), operators.number(0)],
    collapse_operators=[np.sqrt(0.1) * a],
    store_intermediate_results=True,
    integrator=RungeKuttaIntegrator())

get_result = lambda idx, res: [
    exp_vals[idx].expectation() for exp_vals in res.expectation_values()
]

decay_results = [
    get_result(0, evolution_result_decay),
    get_result(1, evolution_result_decay)
]

fig = plt.figure(figsize=(18, 6))

plt.plot(steps, decay_results[0])
plt.plot(steps, decay_results[1])
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.legend(("Cavity Photon Number", "Atom Excitation Probability"))
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig('cavity_qed.png', dpi=fig.dpi)
