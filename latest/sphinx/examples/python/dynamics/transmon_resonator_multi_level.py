import cudaq
from cudaq import operators, ScalarOperator, Schedule, ElementaryOperator, ScipyZvodeIntegrator
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# This example demonstrates a simulation of a superconducting transmon qubit coupled to a resonator (i.e., cavity).
# Number of transmon levels
num_levels = 3

# Number of cavity photons
N = 10

# System dimensions: transmon + cavity
dimensions = {0: num_levels, 1: N}

# Alias for commonly used operators
# Cavity operators
a = operators.annihilate(1)
a_dag = operators.create(1)

# Transmon operators
b = operators.annihilate(0)
b_dag = operators.create(0)

# Model parameters
alpha = 2 * np.pi * 0.3
K = 2 * np.pi * 0.4
delta_c = -2 * np.pi * 0.3
gate_duration = 4.0
sigma = gate_duration / 6.0
gamma_max = 2 * np.pi * 0.3
Omega_max = 2 * np.pi * 0.2
hamiltonian = (alpha / 2.) * b_dag * b_dag * b * b + ScalarOperator(
    lambda t: gamma_max * np.exp(-((t - gate_duration / 2)**2) / 2 / sigma**2)
) * (b + b_dag) + delta_c * a_dag * a + ScalarOperator(lambda t: Omega_max) * (
    a * b_dag + b * a_dag)

# Initial state of the system
# Transmon in a |0>
transmon_state = cp.zeros(dimensions[0], dtype=cp.complex128)
transmon_state[0] = 1.0
# Cavity in a |0>
cavity_state = cp.zeros(dimensions[1], dtype=cp.complex128)
cavity_state[0] = 1.0
psi0 = cudaq.State.from_data(cp.kron(transmon_state, cavity_state))

steps = np.linspace(0, 5, 1000)
schedule = Schedule(steps, ["t"])


# Define projector matrix |state><state|
def projector(dim, state):
    mat = np.zeros((dim, dim), dtype=np.complex128)
    mat[state, state] = 1.0
    return mat


# These are projectors to transmon |0>, |1> and |2> states.
ElementaryOperator.define("P0", [0], lambda dim: projector(dim, 0))
ElementaryOperator.define("P1", [0], lambda dim: projector(dim, 1))
ElementaryOperator.define("P2", [0], lambda dim: projector(dim, 2))
proj0 = ElementaryOperator("P0", [0])
proj1 = ElementaryOperator("P1", [0])
proj2 = ElementaryOperator("P2", [0])

# Evolve the system
evolution_result = cudaq.evolve(hamiltonian,
                                dimensions,
                                schedule,
                                psi0,
                                observables=[proj0, proj1, proj2],
                                collapse_operators=[],
                                store_intermediate_results=True,
                                integrator=ScipyZvodeIntegrator())

get_result = lambda idx, res: [
    exp_vals[idx].expectation() for exp_vals in res.expectation_values()
]
state_results = [
    get_result(0, evolution_result),
    get_result(1, evolution_result),
    get_result(2, evolution_result)
]

fig = plt.figure(figsize=(12, 6))
plt.plot(steps, state_results[0])
plt.plot(steps, state_results[1])
plt.plot(steps, state_results[2])
plt.ylabel("Probability")
plt.xlabel("Time [us]")
plt.legend(("|0>", "|1>", "|2>"))
plt.title("Transmon Qubit Evolution")
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig("transmon_resonator_multi_level.png", dpi=fig.dpi)
