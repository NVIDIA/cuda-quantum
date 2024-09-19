import cudaq
from cudaq.operator import *

import numpy as np
import cupy as cp

cudaq.set_target("nvidia-dynamics")

# Solving the Lindblad dynamics of a qubit chain
# https://qiskit-community.github.io/qiskit-dynamics/tutorials/Lindblad_dynamics_simulation.html

N = 6
nu_z = 4.
nu_x = 1.
J = 4.
Gamma = 4.
qubits = range(N)

dimensions = {}

for i in qubits:
    dimensions[i] = 2

H = operators.zero()

for i in qubits:
    H += .5 * 2 * np.pi * nu_x * pauli.x(i) 
    H += .5 * 2 * np.pi * nu_z * pauli.z(i) 

    if N > 1:
        j = i + 1 if i < (N - 1) else 0  # Nearest neighbors, with periodic boundary conditions
        op = pauli.x(i) * pauli.x(j)
        H += .5 * 2 * np.pi * J * op

        op = pauli.y(i) * pauli.y(j)
        H += .5 * 2 * np.pi * J * op

L_ops = []
for i in qubits:
    L_ops.append(np.sqrt(Gamma) * pauli.plus(i))

t_final = 8. / Gamma
tau = .01
n_steps = int(np.ceil(t_final / tau)) + 1
steps = np.linspace(0., t_final, n_steps)
schedule = Schedule(steps, ["time"])

rho0_= cp.zeros((2**N, 2**N), dtype=cp.complex128)
rho0_[0][0] = 1.0
rho0 = cudaq.State.from_data(rho0_)

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

evolution_result = evolve(H,
                          dimensions,
                          schedule,
                          rho0,
                          observables=[x_mean,
                                       y_mean,
                                       z_mean],
                          collapse_operators=L_ops,
                          store_intermediate_results=True,
                          integrator=ScipyZvodeIntegrator(nsteps=10))

exp_val_x = []
exp_val_y = []
exp_val_z = []

for exp_vals in evolution_result.expectation_values():
    exp_val_x.append(exp_vals[0].expectation())
    exp_val_y.append(exp_vals[1].expectation())
    exp_val_z.append(exp_vals[2].expectation())

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 6))
plt.plot(steps, exp_val_x, label = '$ N^{-1}\sum_i \\langle X_i \\rangle$')
plt.plot(steps, exp_val_y, label = '$ N^{-1}\sum_i \\langle Y_i \\rangle$')
plt.plot(steps, exp_val_z, label = '$ N^{-1}\sum_i \\langle Z_i \\rangle$')
plt.legend(fontsize = 16)
plt.ylabel('Expectation value')
plt.xlabel('Time')
fig.savefig('example_5.png', dpi=fig.dpi)

