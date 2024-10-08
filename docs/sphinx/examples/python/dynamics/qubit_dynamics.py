import cudaq
from cudaq.operator import *
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# Set the target to our dynamics simulator
cudaq.set_target("nvidia-dynamics")

# Qubit Hamiltonian
hamiltonian = 2 * np.pi * 0.1 * pauli.x(0)

# Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
dimensions = {0: 2}

# Initial state of the system (ground state).
rho0 = cudaq.State.from_data(
    cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

# Schedule of time steps.
steps = np.linspace(0, 10, 101)
schedule = Schedule(steps, ["time"])

# Run the simulation.
# First, we run the simulation without any collapse operators (ideal).
evolution_result = evolve(hamiltonian,
                          dimensions,
                          schedule,
                          rho0,
                          observables=[pauli.y(0), pauli.z(0)],
                          collapse_operators=[],
                          store_intermediate_results=True,
                          integrator=RungeKuttaIntegrator())

# Now, run the simulation with qubit decaying due to the presence of a collapse operator.
evolution_result_decay = evolve(hamiltonian,
                                dimensions,
                                schedule,
                                rho0,
                                observables=[pauli.y(0), pauli.z(0)],
                                collapse_operators=[np.sqrt(0.05) * pauli.x(0)],
                                store_intermediate_results=True,
                                integrator=RungeKuttaIntegrator())

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
plt.legend(("Sigma-Y", "Sigma-Z"))
plt.title("No decay")

plt.subplot(1, 2, 2)
plt.plot(steps, decay_results[0])
plt.plot(steps, decay_results[1])
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.legend(("Sigma-Y", "Sigma-Z"))
plt.title("With decay")

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig('qubit_dynamics.png', dpi=fig.dpi)
