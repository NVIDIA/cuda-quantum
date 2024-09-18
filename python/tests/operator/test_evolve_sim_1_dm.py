import cudaq
from cudaq.operator import *
import numpy as np

# Run on a state vector simulator
cudaq.set_target("density-matrix-cpu")

hamiltonian = 2 * np.pi * 0.1 * pauli.x(0)
num_qubits = 1
dimensions = {0: 2}
rho0 = cudaq.State.from_data(
    np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128))
steps = np.linspace(0, 10, 101)
schedule = Schedule(steps, ["time"])
evolution_result = evolve(hamiltonian,
                          dimensions,
                          schedule,
                          rho0,
                          observables=[pauli.y(0), pauli.z(0)],
                          collapse_operators=[np.sqrt(0.05) * pauli.x(0)],
                          store_intermediate_results=True)


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
fig.savefig('example_1_sim_dm.png', dpi=fig.dpi)

