import cudaq
from cudaq import operators, Schedule, ScipyZvodeIntegrator
from cudaq.operator import coherent_state
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# This example demonstrate the creation of a so-called "cat state" (superposition of two coherent states) via the non-linear Kerr effect.

# Number of Fock levels
N = 15
# Kerr non-linearity
chi = 1 * 2 * np.pi

dimensions = {0: N}

a = operators.annihilate(0)
a_dag = operators.create(0)

# Defining the Hamiltonian for the system (non-linear Kerr effect)
hamiltonian = 0.5 * chi * a_dag * a_dag * a * a

# we start with a coherent state with alpha=2.0
# This will evolve into a cat state.
rho0 = cudaq.State.from_data(coherent_state(N, 2.0))

# Choose the end time at which the state evolves to the exact cat state.
steps = np.linspace(0, 0.5 * chi / (2 * np.pi), 51)
schedule = Schedule(steps, ["time"])

# Run the simulation: observe the photon count, position and momentum.
evolution_result = cudaq.evolve(hamiltonian,
                                dimensions,
                                schedule,
                                rho0,
                                observables=[
                                    operators.number(0),
                                    operators.position(0),
                                    operators.momentum(0)
                                ],
                                collapse_operators=[],
                                store_intermediate_results=True,
                                integrator=ScipyZvodeIntegrator())

get_result = lambda idx, res: [
    exp_vals[idx].expectation() for exp_vals in res.expectation_values()
]
photon_count = get_result(0, evolution_result)
position = get_result(1, evolution_result)
momentum = get_result(2, evolution_result)

# The expected cat state: superposition of `|alpla>` and `|-alpha>` coherent states.
expected_state = np.exp(1j * np.pi / 4) * coherent_state(N, -2.0j) + np.exp(
    -1j * np.pi / 4) * coherent_state(N, 2.0j)
expected_state = expected_state / cp.linalg.norm(expected_state)

fig = plt.figure(figsize=(14, 4))

ax = plt.subplot(1, 3, 1)
plt.plot(steps, photon_count)
ax.set_ylim([2, 6])
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.title("n")

plt.subplot(1, 3, 2)
plt.plot(steps, position)
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.title("x")

plt.subplot(1, 3, 3)
plt.plot(steps, momentum)
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.title("p")

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig('cat_state_dynamics.png', dpi=fig.dpi)

# Check that we indeed arrive at the expected state
final_state = evolution_result.final_state()
overlap = final_state.overlap(expected_state)
print(f"Overlap with the expected cat state: {overlap}.")

