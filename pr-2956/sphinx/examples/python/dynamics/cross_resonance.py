import cudaq
from cudaq import spin, Schedule, ScipyZvodeIntegrator
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# This example simulates cross-resonance interactions between superconducting qubits.
# Cross-resonance interaction is key to implementing two-qubit conditional gates, e.g., the CNOT gate.
# Ref: A simple all-microwave entangling gate for fixed-frequency superconducting qubits (Physical Review Letters 107, 080502)
# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Device parameters
# Detuning between two qubits
delta = 100 * 2 * np.pi
# Static coupling between qubits
J = 7 * 2 * np.pi
# spurious electromagnetic `crosstalk` due to stray electromagnetic coupling in the device circuit and package
# see (Physical Review Letters 107, 080502)
m_12 = 0.2
# Drive strength
Omega = 20 * 2 * np.pi

# Qubit Hamiltonian (in the rotating frame w.r.t. the target qubit)
hamiltonian = delta / 2 * spin.z(0) + J * (
    spin.minus(1) * spin.plus(0) +
    spin.plus(1) * spin.minus(0)) + Omega * spin.x(0) + m_12 * Omega * spin.x(1)

# Dimensions of sub-system
dimensions = {0: 2, 1: 2}

# Two initial states: |00> and |10>.
# We show the 'conditional' evolution when controlled qubit is in |1> state.
psi_00 = cudaq.State.from_data(
    cp.array([1.0, 0.0, 0.0, 0.0], dtype=cp.complex128))
psi_10 = cudaq.State.from_data(
    cp.array([0.0, 1.0, 0.0, 0.0], dtype=cp.complex128))

# Schedule of time steps.
steps = np.linspace(0.0, 1.0, 1001)
schedule = Schedule(steps, ["time"])

# Run the simulations (batched).
evolution_results = cudaq.evolve(hamiltonian,
                                 dimensions,
                                 schedule, [psi_00, psi_10],
                                 observables=[
                                     spin.x(0),
                                     spin.y(0),
                                     spin.z(0),
                                     spin.x(1),
                                     spin.y(1),
                                     spin.z(1)
                                 ],
                                 collapse_operators=[],
                                 store_intermediate_results=True,
                                 integrator=ScipyZvodeIntegrator())

get_result = lambda idx, res: [
    exp_vals[idx].expectation() for exp_vals in res.expectation_values()
]
results_00 = [get_result(i, evolution_results[0]) for i in range(6)]
results_10 = [get_result(i, evolution_results[1]) for i in range(6)]

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
