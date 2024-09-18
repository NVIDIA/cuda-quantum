import cudaq
from cudaq.operator import *

import numpy as np
import cupy as cp

cudaq.set_target("nvidia-dynamics")


# Time-dependent Hamiltonian
# Square-pulse
def square_pulse(t):
    # print("Callback @ t =", t)
    if (t >= 2) & (t <= 4):
        return 1.0
    else:
        return 0.0


squared_pulse = ScalarOperator(square_pulse)
hamiltonian = squared_pulse * pauli.x(0)
dimensions = {0: 2}
rho0 = cudaq.State.from_data(
    cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

steps = np.linspace(0, 10, 101)
schedule = Schedule(steps, ["time"])

evolution_result = evolve(hamiltonian,
                          dimensions,
                          schedule,
                          rho0,
                          observables=[operators.number(0)],
                          collapse_operators=[],
                          store_intermediate_results=True,
                          integrator=ScipyZvodeIntegrator(nsteps=10))

pulse_values = [square_pulse(t) for t in steps]
exp_val_excitation = []

for exp_vals in evolution_result.expectation_values():
    exp_val_excitation.append(exp_vals[0].expectation())


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 6))
plt.plot(steps, exp_val_excitation)
plt.plot(steps, pulse_values)
plt.ylabel('Expectation value')
plt.xlabel('Time')
plt.legend(("Population in |1>", "Drive Pulse"))
fig.savefig('example_3.png', dpi=fig.dpi)
