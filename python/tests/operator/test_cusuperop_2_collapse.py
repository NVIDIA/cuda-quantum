import cudaq
from cudaq.operator import *
import numpy as np
import cupy as cp

cudaq.set_target("nvidia-dynamics")

# https://qutip.readthedocs.io/en/qutip-5.0.x/guide/dynamics/dynamics-master.html#non-unitary-evolution
dimensions = {0: 2, 1: 10}
a = operators.annihilate(1)
a_dag = operators.create(1)
sm = operators.annihilate(0)
sm_dag = operators.create(0)

hamiltonian = 2 * np.pi * operators.number(1) + 2 * np.pi * operators.number(
    0) + 2 * np.pi * 0.25 * (sm * a_dag + sm_dag * a)
qubit_state = cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128)
cavity_state = cp.zeros((10, 10), dtype=cp.complex128)
cavity_state[5][5] = 1.0
rho0 = cudaq.State.from_data(
    cp.kron(qubit_state, cavity_state).reshape((2, 10, 2, 10)))

steps = np.linspace(0, 10, 201)
schedule = Schedule(steps, ["time"])

evolution_result = evolve(
    hamiltonian,
    dimensions,
    schedule,
    rho0,
    observables=[operators.number(1), operators.number(0)],
    collapse_operators=[np.sqrt(0.1) * a],
    store_intermediate_results=False)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 6))
plt.plot(steps, evolution_result.expect[0])
plt.plot(steps, evolution_result.expect[1])
plt.ylabel('Expectation value')
plt.xlabel('Time')
plt.legend(("Cavity Photon Number", "Atom Excitation Probability"))
fig.savefig('example_2_c.png', dpi=fig.dpi)
