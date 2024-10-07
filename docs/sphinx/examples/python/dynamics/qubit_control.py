import cudaq
from cudaq.operator import *
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# This example simulates time evolution of a qubit (`transmon`) being driven close to resonance in the presence of noise (decoherence). 
# Thus, it exhibits Rabi oscillations.

# Set the target to our dynamics simulator
cudaq.set_target("nvidia-dynamics")

# Device parameters
# Qubit resonant frequency 
nu_z = 10.0
# Transverse term
nu_x = 1.0
# Harmonic driving frequency
# Note: we chose a frequency slightly different from the resonant frequency to demonstrate the off-resonance effect.
nu_d = 9.98 


# Qubit Hamiltonian
hamiltonian = 0.5 * 2 * np.pi * nu_z * pauli.z(0) 
# Add modulated driving term to the Hamiltonian
hamiltonian += 2 * np.pi * nu_x * ScalarOperator(lambda t: np.cos(2 * np.pi * nu_d * t)) * pauli.x(0)

# Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
dimensions = {0: 2}

# Initial state of the system (ground state).
rho0 = cudaq.State.from_data(
    cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

# Schedule of time steps.
t_final = 0.5 / nu_x
tau = .005
n_steps = int(np.ceil(t_final / tau)) + 1
steps1 = np.linspace(0, t_final, n_steps)
schedule = Schedule(steps1, ["t"])

# Run the simulation.
# First, we run the simulation without any collapse operators (no decoherence).
evolution_result = evolve(hamiltonian,
                          dimensions,
                          schedule,
                          rho0,
                          observables=[pauli.x(0), pauli.y(0), pauli.z(0)],
                          collapse_operators=[],
                          store_intermediate_results=True,
                          integrator=ScipyZvodeIntegrator())

# Now, run the simulation with qubit decoherence
Gamma_1 = 0.8
Gamma_2 = 0.2
# Use a different time scale in this case to demonstrate the effect of decoherence.
t_final = 5.5 / max(Gamma_1, Gamma_2)
n_steps = int(np.ceil(t_final / tau)) + 1
steps2 = np.linspace(0, t_final, n_steps)
schedule = Schedule(steps2, ["t"])

evolution_result_decay = evolve(hamiltonian,
                          dimensions,
                          schedule,
                          rho0,
                          observables=[pauli.x(0), pauli.y(0), pauli.z(0)],
                          collapse_operators=[np.sqrt(Gamma_1) * pauli.plus(0), np.sqrt(Gamma_2) * pauli.z(0)],
                          store_intermediate_results=True,
                          integrator=ScipyZvodeIntegrator())

get_result = lambda idx, res: [exp_vals[idx].expectation() for exp_vals in res.expectation_values()]
ideal_results = [get_result(0, evolution_result), get_result(1, evolution_result), get_result(2, evolution_result)]
decoherence_results = [get_result(0, evolution_result_decay), get_result(1, evolution_result_decay), get_result(2, evolution_result_decay)]

fig = plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(steps1, ideal_results[0])
plt.plot(steps1, ideal_results[1])
plt.plot(steps1, ideal_results[2])
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.legend(("Sigma-X", "Sigma-Y", "Sigma-Z"))
plt.title("No decoherence")

plt.subplot(1, 2, 2)
plt.plot(steps2, decoherence_results[0])
plt.plot(steps2, decoherence_results[1])
plt.plot(steps2, decoherence_results[2])
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.legend(("Sigma-X", "Sigma-Y", "Sigma-Z"))
plt.title("With decoherence")

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig('qubit_control.png', dpi=fig.dpi)