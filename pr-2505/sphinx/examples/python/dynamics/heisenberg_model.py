import cudaq
from cudaq import operators, spin, Schedule, ScipyZvodeIntegrator

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# In this example, we solve the Quantum Heisenberg model (https://en.wikipedia.org/wiki/Quantum_Heisenberg_model),
# which exhibits the so-called quantum quench effect.
# e.g., see `Quantum quenches in the anisotropic spin-1/2 Heisenberg chain: different approaches to many-body dynamics far from equilibrium`
# (New J. Phys. 12 055017)
# Number of spins
N = 9
dimensions = {}
for i in range(N):
    dimensions[i] = 2

# Initial state: alternating spin up and down
spin_state = ''
for i in range(N):
    spin_state += str(int(i % 2))

# Observable is the staggered magnetization operator
staggered_magnetization_op = operators.zero()
for i in range(N):
    if i % 2 == 0:
        staggered_magnetization_op += spin.z(i)
    else:
        staggered_magnetization_op -= spin.z(i)

staggered_magnetization_op /= N

observe_results = []
for g in [0.0, 0.25, 4.0]:
    # Heisenberg model spin coupling strength
    Jx = 1.0
    Jy = 1.0
    Jz = g

    # Construct the Hamiltonian
    H = operators.zero()

    for i in range(N - 1):
        H += Jx * spin.x(i) * spin.x(i + 1)
        H += Jy * spin.y(i) * spin.y(i + 1)
        H += Jz * spin.z(i) * spin.z(i + 1)

    steps = np.linspace(0.0, 5, 1000)
    schedule = Schedule(steps, ["time"])

    # Prepare the initial state vector
    psi0_ = cp.zeros(2**N, dtype=cp.complex128)
    psi0_[int(spin_state, 2)] = 1.0
    psi0 = cudaq.State.from_data(psi0_)

    # Run the simulation
    evolution_result = cudaq.evolve(H,
                                    dimensions,
                                    schedule,
                                    psi0,
                                    observables=[staggered_magnetization_op],
                                    collapse_operators=[],
                                    store_intermediate_results=True,
                                    integrator=ScipyZvodeIntegrator())

    exp_val = [
        exp_vals[0].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]

    observe_results.append((g, exp_val))

# Plot the results
fig = plt.figure(figsize=(12, 6))
for g, exp_val in observe_results:
    plt.plot(steps, exp_val, label=f'$ g = {g}$')
plt.legend(fontsize=16)
plt.ylabel("Staggered Magnetization")
plt.xlabel("Time")
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig("heisenberg_model.png", dpi=fig.dpi)
