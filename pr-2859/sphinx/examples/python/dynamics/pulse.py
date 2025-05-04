import cudaq
from cudaq import spin, boson, ScalarOperator, Schedule, ScipyZvodeIntegrator
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# This example simulates time evolution of a qubit (`transmon`) being driven by a pulse.
# The pulse is a modulated signal with a Gaussian envelop.
# The simulation is performed in the 'lab' frame.

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Device parameters
# Strength of the Rabi-rate in GHz.
rabi_rate = 0.1

# Frequency of the qubit transition in GHz.
omega = 5.0 * 2 * np.pi

# Define Gaussian envelope function to approximately implement a `rx(pi/2)` gate.
amplitude = 1. / 2.0  # Pi/2 rotation
sigma = 1.0 / rabi_rate / amplitude
pulse_duration = 6 * sigma


def gaussian(t, duration, amplitude, sigma):
    # Gaussian envelope function
    return amplitude * np.exp(-0.5 * (t - duration / 2)**2 / (sigma)**2)


def signal(t):
    # Modulated signal
    return np.cos(omega * t) * gaussian(t, pulse_duration, amplitude, sigma)


# Qubit Hamiltonian
hamiltonian = omega * spin.z(0) / 2
# Add modulated driving term to the Hamiltonian
hamiltonian += np.pi * rabi_rate * ScalarOperator(signal) * spin.x(0)

# Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
dimensions = {0: 2}

# Initial state of the system (ground state).
psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

# Schedule of time steps.
# Since this is a lab-frame simulation, the time step must be small to accurately capture the modulated signal.
dt = 1 / omega / 20
n_steps = int(np.ceil(pulse_duration / dt)) + 1
steps = np.linspace(0, pulse_duration, n_steps)
schedule = Schedule(steps, ["t"])

# Run the simulation.
# First, we run the simulation without any collapse operators (no decoherence).
evolution_result = cudaq.evolve(hamiltonian,
                                dimensions,
                                schedule,
                                psi0,
                                observables=[boson.number(0)],
                                collapse_operators=[],
                                store_intermediate_results=True,
                                integrator=ScipyZvodeIntegrator())

pop1 = [
    exp_vals[0].expectation()
    for exp_vals in evolution_result.expectation_values()
]
pop0 = [1.0 - x for x in pop1]
fig = plt.figure(figsize=(6, 16))
envelop = [gaussian(t, pulse_duration, amplitude, sigma) for t in steps]

plt.subplot(3, 1, 1)
plt.plot(steps, envelop)
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Envelope")

modulated_signal = [
    np.cos(omega * t) * gaussian(t, pulse_duration, amplitude, sigma)
    for t in steps
]
plt.subplot(3, 1, 2)
plt.plot(steps, modulated_signal)
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Signal")

plt.subplot(3, 1, 3)
plt.plot(steps, pop0)
plt.plot(steps, pop1)
plt.ylabel("Population")
plt.xlabel("Time")
plt.legend(("Population in |0>", "Population in |1>"))
plt.title("Qubit State")

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
fig.savefig("pulse.png", dpi=fig.dpi)
