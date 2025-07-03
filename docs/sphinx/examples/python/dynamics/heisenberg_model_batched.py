import cudaq
from cudaq import spin, Schedule, ScipyZvodeIntegrator

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

# Specifically, we demonstrate the use of batched Hamiltonian operators to simulate the Heisenberg model
# with different coupling strengths. 
# These batched Hamiltonian operators allow us to efficiently compute the dynamics
# for multiple Hamiltonian operators in a single simulation run.
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
staggered_magnetization_op = spin.empty()
for i in range(N):
    if i % 2 == 0:
        staggered_magnetization_op += spin.z(i)
    else:
        staggered_magnetization_op -= spin.z(i)

staggered_magnetization_op /= N

observe_results = []
batched_hamiltonian = []
for g in [0.0, 0.25, 4.0]:
    # Heisenberg model spin coupling strength
    Jx = 1.0
    Jy = 1.0
    Jz = g

    # Construct the Hamiltonian
    H = spin.empty()

    for i in range(N - 1):
        H += Jx * spin.x(i) * spin.x(i + 1)
        H += Jy * spin.y(i) * spin.y(i + 1)
        H += Jz * spin.z(i) * spin.z(i + 1)
    # Append the Hamiltonian to the batched list
    batched_hamiltonian.append(H)

steps = np.linspace(0.0, 5, 1000)
schedule = Schedule(steps, ["time"])

# Prepare the initial state vector
psi0_ = cp.zeros(2**N, dtype=cp.complex128)
psi0_[int(spin_state, 2)] = 1.0
psi0 = cudaq.State.from_data(psi0_)

# Batched initial states (all the same in this case)
batched_psi0 = [psi0] * len(batched_hamiltonian)

# Run the simulation in batched mode
evolution_results = cudaq.evolve(batched_hamiltonian,
                                dimensions,
                                schedule,
                                batched_psi0,
                                observables=[staggered_magnetization_op],
                                collapse_operators=[],
                                store_intermediate_results=cudaq.
                                IntermediateResultSave.EXPECTATION_VALUE,
                                integrator=ScipyZvodeIntegrator())

for g, evolution_result in zip([0.0, 0.25, 4.0], evolution_results):
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
fig.savefig("heisenberg_model_batched.png", dpi=fig.dpi)
