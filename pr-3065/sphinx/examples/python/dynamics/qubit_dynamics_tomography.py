import cudaq
from cudaq import spin, Schedule, RungeKuttaIntegrator
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Qubit Hamiltonian
hamiltonian = 2 * np.pi * 0.1 * spin.x(0)

# Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
dimensions = {0: 2}

# Initial states in the `SIC-POVM` set: https://en.wikipedia.org/wiki/SIC-POVM
psi_1 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))
psi_2 = cudaq.State.from_data(
    cp.array([1.0 / np.sqrt(3.0), np.sqrt(2.0 / 3.0)], dtype=cp.complex128))
psi_3 = cudaq.State.from_data(
    cp.array([
        1.0 / np.sqrt(3.0),
        np.sqrt(2.0 / 3.0) * np.exp(1j * 2.0 * np.pi / 3.0)
    ],
             dtype=cp.complex128))
psi_4 = cudaq.State.from_data(
    cp.array([
        1.0 / np.sqrt(3.0),
        np.sqrt(2.0 / 3.0) * np.exp(1j * 4.0 * np.pi / 3.0)
    ],
             dtype=cp.complex128))

# We run the evolution for all the SIC state to determine the process tomography.
sic_states = [psi_1, psi_2, psi_3, psi_4]
# Schedule of time steps.
steps = np.linspace(0, 10, 101)
schedule = Schedule(steps, ["time"])

# Run the simulation.
evolution_results = cudaq.evolve(hamiltonian,
                                 dimensions,
                                 schedule,
                                 sic_states,
                                 observables=[spin.x(0),
                                              spin.y(0),
                                              spin.z(0)],
                                 collapse_operators=[],
                                 store_intermediate_results=True,
                                 integrator=RungeKuttaIntegrator())

get_result = lambda idx, res: [
    exp_vals[idx].expectation() for exp_vals in res.expectation_values()
]
tomography_results = [[
    get_result(0, evolution_result),
    get_result(1, evolution_result),
    get_result(2, evolution_result)
] for evolution_result in evolution_results]

fig = plt.figure(figsize=(18, 12))
for i in range(len(tomography_results)):
    plt.subplot(2, 2, i + 1)
    plt.plot(steps, tomography_results[i][0])
    plt.plot(steps, tomography_results[i][1])
    plt.plot(steps, tomography_results[i][2])
    plt.ylim(-1.0, 1.0)
    plt.ylabel("Expectation value")
    plt.xlabel("Time")
    plt.legend(("Sigma-X", "Sigma-Y", "Sigma-Z"))
    plt.title(f"SIC state {i}")

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig('qubit_dynamics_tomography.png', dpi=fig.dpi)
