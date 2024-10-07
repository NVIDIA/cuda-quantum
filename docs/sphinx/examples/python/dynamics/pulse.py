import cudaq
from cudaq.operator import *
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# This example simulates time evolution of a qubit (`transmon`) being driven by a pulse.
# The pulse is a modulated signal with a gaussian envelop.

# Set the target to our dynamics simulator
cudaq.set_target("nvidia-dynamics")

# Device parameters
# Strength of the Rabi-rate in GHz.
r = 0.1

# Frequency of the qubit transition in GHz.
w = 5.

# Sample rate of the backend in ns.
dt = 1 / 4.5

# Define gaussian envelope function to approximately implement an sx gate.
amp = 1. / 2.0
sig = 1.0 / r / amp
T = 6 * sig


def gaussian(t, duration, amp, sigma):
    return amp * np.exp(-0.5 * (t - duration / 2)**2 / (sigma)**2)


def signal(t):
    # Modulated signal
    return np.cos(2 * np.pi * w * t) * gaussian(t, T, amp, sig)


# Qubit Hamiltonian
hamiltonian = 2 * np.pi * w * pauli.z(0) / 2
# Add modulated driving term to the Hamiltonian
hamiltonian += np.pi * r * ScalarOperator(signal) * pauli.x(0)

# Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
dimensions = {0: 2}

# Initial state of the system (ground state).
psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

# Schedule of time steps.
t_final = T
tau = dt / 100
n_steps = int(np.ceil(t_final / tau)) + 1
steps = np.linspace(0, t_final, n_steps)
schedule = Schedule(steps, ["t"])

# Run the simulation.
# First, we run the simulation without any collapse operators (no decoherence).
evolution_result = evolve(hamiltonian,
                          dimensions,
                          schedule,
                          psi0,
                          observables=[],
                          collapse_operators=[],
                          store_intermediate_results=True,
                          integrator=ScipyZvodeIntegrator())

pop0 = [np.abs(psi[0])**2 for psi in evolution_result.intermediate_states()]
pop1 = [np.abs(psi[1])**2 for psi in evolution_result.intermediate_states()]

fig = plt.figure(figsize=(6, 16))
envelop = [gaussian(t, T, amp, sig) for t in steps]

plt.subplot(3, 1, 1)
plt.plot(steps, envelop)
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Envelope")

modulated = [
    np.cos(2 * np.pi * w * t) * gaussian(t, T, amp, sig) for t in steps
]
plt.subplot(3, 1, 2)
plt.plot(steps, modulated)
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
