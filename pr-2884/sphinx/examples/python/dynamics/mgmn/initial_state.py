import cudaq
from cudaq import spin, Schedule, RungeKuttaIntegrator

import numpy as np
import matplotlib.pyplot as plt
import os

# On a system with multiple GPUs, `mpiexec` can be used as follows:
# `mpiexec -np <N> python3 multi_gpu.py `
cudaq.mpi.initialize()

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Large number of spins
N = 20
dimensions = {}
for i in range(N):
    dimensions[i] = 2

# Observable is the average magnetization operator
avg_magnetization_op = spin.empty()
for i in range(N):
    avg_magnetization_op += (spin.z(i) / N)

# Arbitrary coupling constant
g = 1.0
# Construct the Hamiltonian
H = spin.empty()
for i in range(N):
    H += 2 * np.pi * spin.x(i)
    H += 2 * np.pi * spin.y(i)
for i in range(N - 1):
    H += 2 * np.pi * g * spin.x(i) * spin.x(i + 1)
    H += 2 * np.pi * g * spin.y(i) * spin.z(i + 1)

steps = np.linspace(0.0, 1, 200)
schedule = Schedule(steps, ["time"])

# Initial state (expressed as an enum)
psi0 = cudaq.dynamics.InitialState.ZERO
# This can also be used to initialize a uniformly-distributed wave-function instead.
# `psi0 = cudaq.dynamics.InitialState.UNIFORM`

# Run the simulation
evolution_result = cudaq.evolve(H,
                                dimensions,
                                schedule,
                                psi0,
                                observables=[avg_magnetization_op],
                                collapse_operators=[],
                                store_intermediate_results=True,
                                integrator=RungeKuttaIntegrator())

exp_val = [
    exp_vals[0].expectation()
    for exp_vals in evolution_result.expectation_values()
]

if cudaq.mpi.rank() == 0:
    # Plot the results
    fig = plt.figure(figsize=(12, 6))
    plt.plot(steps, exp_val)
    plt.ylabel("Average Magnetization")
    plt.xlabel("Time")
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    fig.savefig("spin_model.png", dpi=fig.dpi)

cudaq.mpi.finalize()
