import cudaq
from cudaq.operator import *

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os

# Set the target to our dynamics simulator
cudaq.set_target("nvidia-dynamics")

# In this example, we solve the Lindblad dynamics of qubits interacting with their nearest neighbors along a ring.

# Number of qubits
N = 6

# Qubits are modeled as simple `transmons`.
# Qubit resonance frequency
nu_z = 4.0

# Transverse term amplitude
nu_x = 1.0

# Coupling strength
J = 4.0

# Decoherence rate
Gamma = 4.0
qubits = range(N)
dimensions = {}
for i in qubits:
    dimensions[i] = 2

# Construct the Hamiltonian
H = operators.zero()

for i in qubits:
    H += .5 * 2 * np.pi * nu_x * pauli.x(i)
    H += .5 * 2 * np.pi * nu_z * pauli.z(i)

    if N > 1:
        j = i + 1 if i < (
            N -
            1) else 0  # Nearest neighbors, with periodic boundary conditions
        H += .5 * 2 * np.pi * J * pauli.x(i) * pauli.x(j)
        H += .5 * 2 * np.pi * J * pauli.y(i) * pauli.y(j)

# Collapse operators
L_ops = []
for i in qubits:
    L_ops.append(np.sqrt(Gamma) * pauli.plus(i))

t_final = 8. / Gamma
tau = .01
n_steps = int(np.ceil(t_final / tau)) + 1
steps = np.linspace(0., t_final, n_steps)
schedule = Schedule(steps, ["time"])

# Initial state
rho0_ = cp.zeros((2**N, 2**N), dtype=cp.complex128)
rho0_[0][0] = 1.0
rho0 = cudaq.State.from_data(rho0_)

# Observables: mean field values
x_mean = operators.zero()
y_mean = operators.zero()
z_mean = operators.zero()

for i in qubits:
    x_mean += pauli.x(i)
    y_mean += pauli.y(i)
    z_mean += pauli.z(i)

x_mean /= N
y_mean /= N
z_mean /= N

# Run the simulation
evolution_result = evolve(H,
                          dimensions,
                          schedule,
                          rho0,
                          observables=[x_mean, y_mean, z_mean],
                          collapse_operators=L_ops,
                          store_intermediate_results=True,
                          integrator=ScipyZvodeIntegrator())

exp_val_x = [
    exp_vals[0].expectation()
    for exp_vals in evolution_result.expectation_values()
]
exp_val_y = [
    exp_vals[1].expectation()
    for exp_vals in evolution_result.expectation_values()
]
exp_val_z = [
    exp_vals[2].expectation()
    for exp_vals in evolution_result.expectation_values()
]

fig = plt.figure(figsize=(12, 6))
plt.plot(steps, exp_val_x, label='$ N^{-1}\sum_i \\langle X_i \\rangle$')
plt.plot(steps, exp_val_y, label='$ N^{-1}\sum_i \\langle Y_i \\rangle$')
plt.plot(steps, exp_val_z, label='$ N^{-1}\sum_i \\langle Z_i \\rangle$')
plt.legend(fontsize=16)
plt.ylabel("Expectation value")
plt.xlabel("Time")
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig("qubit_chain.png", dpi=fig.dpi)
