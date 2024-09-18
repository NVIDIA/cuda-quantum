import cudaq
from cudaq.operator import *
import numpy as np
import cupy as cp

cudaq.set_target("nvidia-dynamics")

hamiltonian = 2 * np.pi * 0.1 * pauli.x(0)
num_qubits = 1
dimensions = {0: 2}
rho0 = cudaq.State.from_data(
    cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))
steps = np.linspace(0, 10, 101)
schedule = Schedule(steps, ["time"])
evolution_result = evolve(hamiltonian,
                          dimensions,
                          schedule,
                          rho0,
                          observables=[pauli.y(0), pauli.z(0)],
                          collapse_operators=[],
                          store_intermediate_results=True,
                          integrator=RungeKuttaIntegrator(nsteps=10))

exp_val_y = []
exp_val_z = []

for exp_vals in evolution_result.expectation_values():
    exp_val_y.append(exp_vals[0].expectation())
    exp_val_z.append(exp_vals[1].expectation())

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 6))
plt.plot(steps, exp_val_y)
plt.plot(steps, exp_val_z)
plt.ylabel('Expectation value')
plt.xlabel('Time')
plt.legend(("Sigma-Y", "Sigma-Z"))
fig.savefig('example_1.png', dpi=fig.dpi)

for state in evolution_result.intermediate_states():
    state.dump()
