import cudaq
from cudaq import spin, ScalarOperator, Schedule, ScipyZvodeIntegrator
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# This example simulates time evolution of a qubit (`transmon`) being driven close to resonance in the presence of noise (decoherence).
# Thus, it exhibits Rabi oscillations.
# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Qubit Hamiltonian reference: https://qiskit-community.github.io/qiskit-dynamics/tutorials/Rabi_oscillations.html
# Device parameters
# Qubit resonant frequency
omega_z = 10.0 * 2 * np.pi
# Transverse term
omega_x = 2 * np.pi
# Harmonic driving frequency
# Note: we chose a frequency slightly different from the resonant frequency to demonstrate the off-resonance effect.
omega_drive = 0.99 * omega_z

# Qubit Hamiltonian
hamiltonian = 0.5 * omega_z * spin.z(0)
# Add modulated driving term to the Hamiltonian
hamiltonian += omega_x * ScalarOperator(
    lambda t: np.cos(omega_drive * t)) * spin.x(0)

# Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
dimensions = {0: 2}

# Initial state of the system (ground state).
rho0 = cudaq.State.from_data(
    cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

# Schedule of time steps.
t_final = np.pi / omega_x
dt = 2.0 * np.pi / omega_drive / 100
n_steps = int(np.ceil(t_final / dt)) + 1
steps = np.linspace(0, t_final, n_steps)
schedule = Schedule(steps, ["t"])

# Run the simulation.
# First, we run the simulation without any collapse operators (no decoherence).
evolution_result = cudaq.evolve(hamiltonian,
                                dimensions,
                                schedule,
                                rho0,
                                observables=[spin.x(0),
                                             spin.y(0),
                                             spin.z(0)],
                                collapse_operators=[],
                                store_intermediate_results=True,
                                integrator=ScipyZvodeIntegrator())

# Now, run the simulation with qubit decoherence
gamma_sm = 4.0
gamma_sz = 1.0
evolution_result_decay = cudaq.evolve(
    hamiltonian,
    dimensions,
    schedule,
    rho0,
    observables=[spin.x(0), spin.y(0), spin.z(0)],
    collapse_operators=[
        np.sqrt(gamma_sm) * spin.plus(0),
        np.sqrt(gamma_sz) * spin.z(0)
    ],
    store_intermediate_results=True,
    integrator=ScipyZvodeIntegrator())

get_result = lambda idx, res: [
    exp_vals[idx].expectation() for exp_vals in res.expectation_values()
]
ideal_results = [
    get_result(0, evolution_result),
    get_result(1, evolution_result),
    get_result(2, evolution_result)
]
decoherence_results = [
    get_result(0, evolution_result_decay),
    get_result(1, evolution_result_decay),
    get_result(2, evolution_result_decay)
]

fig = plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, ideal_results[0])
plt.plot(steps, ideal_results[1])
plt.plot(steps, ideal_results[2])
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.legend(("Sigma-X", "Sigma-Y", "Sigma-Z"))
plt.title("No decoherence")

plt.subplot(1, 2, 2)
plt.plot(steps, decoherence_results[0])
plt.plot(steps, decoherence_results[1])
plt.plot(steps, decoherence_results[2])
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.legend(("Sigma-X", "Sigma-Y", "Sigma-Z"))
plt.title("With decoherence")

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig('qubit_control.png', dpi=fig.dpi)
