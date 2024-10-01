import cudaq
from cudaq.operator import *
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# This example simulates cross-resonance interactions between superconducting qubits.
# Cross-resonance interaction is key to implementing two-qubit conditional gates, e.g., the CNOT gate.

# Set the target to our dynamics simulator
cudaq.set_target("nvidia-dynamics")

# Device parameters
# Detuning between two qubits
delta = 100 * 2 * np.pi
# Static coupling between qubits
J = 7 * 2 * np.pi
# spurious electromagnetic crosstalk
m2 = 0.2
# Drive strength
Omega = 20 * 2 * np.pi

# Qubit Hamiltonian
hamiltonian = delta / 2 * pauli.z(0) + J * (
    pauli.minus(1) * pauli.plus(0) + pauli.plus(1) *
    pauli.minus(0)) + Omega * pauli.x(0) + m2 * Omega * pauli.x(1)

# Dimensions of sub-system
dimensions = {0: 2, 1: 2}

# Initial state of the system (ground state).
rho0 = cudaq.State.from_data(
    cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

# Two initial states: |00> and |10>.
# We show the 'conditional' evolution when controlled qubit is in |1> state.
psi_00 = cudaq.State.from_data(
    cp.array([1.0, 0.0, 0.0, 0.0], dtype=cp.complex128))
psi_10 = cudaq.State.from_data(
    cp.array([0.0, 0.0, 1.0, 0.0], dtype=cp.complex128))

# Schedule of time steps.
steps = np.linspace(0.0, 1.0, 1001)
schedule = Schedule(steps, ["time"])

# Run the simulation.
# Control bit = 0
evolution_result_00 = evolve(hamiltonian,
                             dimensions,
                             schedule,
                             psi_00,
                             observables=[
                                 pauli.x(0),
                                 pauli.y(0),
                                 pauli.z(0),
                                 pauli.x(1),
                                 pauli.y(1),
                                 pauli.z(1)
                             ],
                             collapse_operators=[],
                             store_intermediate_results=True,
                             integrator=ScipyZvodeIntegrator())

# Control bit = 1
evolution_result_10 = evolve(hamiltonian,
                             dimensions,
                             schedule,
                             psi_10,
                             observables=[
                                 pauli.x(0),
                                 pauli.y(0),
                                 pauli.z(0),
                                 pauli.x(1),
                                 pauli.y(1),
                                 pauli.z(1)
                             ],
                             collapse_operators=[],
                             store_intermediate_results=True,
                             integrator=ScipyZvodeIntegrator())

get_result = lambda idx, res: [
    exp_vals[idx].expectation() for exp_vals in res.expectation_values()
]
results_00 = [
    get_result(0, evolution_result_00),
    get_result(1, evolution_result_00),
    get_result(2, evolution_result_00),
    get_result(3, evolution_result_00),
    get_result(4, evolution_result_00),
    get_result(5, evolution_result_00)
]
results_10 = [
    get_result(0, evolution_result_10),
    get_result(1, evolution_result_10),
    get_result(2, evolution_result_10),
    get_result(3, evolution_result_10),
    get_result(4, evolution_result_10),
    get_result(5, evolution_result_10)
]

# The changes in recession frequencies of the target qubit when control qubit is in |1> state allow us to implement two-qubit conditional gates.
fig = plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, results_00[5])
plt.plot(steps, results_10[5])
plt.ylabel(r"$\langle Z_2\rangle$")
plt.xlabel("Time")
plt.legend((r"$|\psi_0\rangle=|00\rangle$", r"$|\psi_0\rangle=|10\rangle$"))

plt.subplot(1, 2, 2)
plt.plot(steps, results_00[4])
plt.plot(steps, results_10[4])
plt.ylabel(r"$\langle Y_2\rangle$")
plt.xlabel("Time")
plt.legend((r"$|\psi_0\rangle=|00\rangle$", r"$|\psi_0\rangle=|10\rangle$"))

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig('cross_resonance.png', dpi=fig.dpi)
