import cudaq
from cudaq.operator import *
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt

# This example demonstrates the use of the dynamics simulator and optimizer to optimize a pulse.
# Set the target to our dynamics simulator
cudaq.set_target("nvidia-dynamics")

# Device parameters
delta = 0.0  # Detuning of the drive
alpha = -340.0  # Anharmonicity
sigma = 0.01  # sigma of the Gaussian pulse
cutoff = 4.0 * sigma  # total length of drive pulse

# Dimensions of sub-system
# We model transmon as a 3-level system to account for leakage.
dimensions = {0: 3}

# Initial state of the system (ground state).
psi0 = cudaq.State.from_data(cp.array([1.0, 0.0, 0.0], dtype=cp.complex128))


def gaussian(t):
    """
    Gaussian shape with cutoff. Starts at t = 0, amplitude normalized to one
    """
    val = (np.exp(-((t-cutoff/2)/sigma)**2/2)-np.exp(-(cutoff/sigma)**2/8)) \
           / (1-np.exp(-(cutoff/sigma)**2/8))
    return val


def dgaussian(t):
    """
    Derivative of Gaussian. Starts at t = 0, amplitude normalized to one
    """
    return -(t - cutoff / 2) / sigma * np.exp(-(
        (t - cutoff / 2) / sigma)**2 / 2 + 0.5)


# Schedule of time steps.
steps = np.linspace(0.0, cutoff, 201)
schedule = Schedule(steps, ["t"])

# We optimize for a X(pi/2) rotation
target_state = np.array([1.0 / np.sqrt(2), -1j / np.sqrt(2), 0.0],
                        dtype=cp.complex128)


# Optimize the amplitude of the drive pulse (DRAG - Derivative Removal by Adiabatic Gate)
def cost_function(amps):
    amplitude = 100 * amps[0]
    drag_amp = 100 * amps[1]
    # Qubit Hamiltonian
    hamiltonian = delta * operators.number(0) + (
        alpha / 2) * operators.create(0) * operators.create(
            0) * operators.annihilate(0) * operators.annihilate(0)
    # Drive term
    hamiltonian += amplitude * ScalarOperator(gaussian) * (
        operators.create(0) + operators.annihilate(0))

    # Drag term (leakage reduction)
    hamiltonian += 1j * drag_amp * ScalarOperator(dgaussian) * (
        operators.annihilate(0) - operators.create(0))

    # We optimize for a X(pi/2) rotation
    evolution_result = evolve(hamiltonian,
                              dimensions,
                              schedule,
                              psi0,
                              observables=[],
                              collapse_operators=[],
                              store_intermediate_results=False,
                              integrator=ScipyZvodeIntegrator())
    final_state = evolution_result.final_state()

    overlap = np.abs(final_state.overlap(target_state))
    print(
        f"Gaussian amplitude = {amplitude}, derivative amplitude = {drag_amp}, Overlap: {overlap}"
    )
    return 1.0 - overlap


# Specify the optimizer
optimizer = cudaq.optimizers.NelderMead()
optimal_error, optimal_parameters = optimizer.optimize(dimensions=2,
                                                       function=cost_function)

print("optimal overlap =", 1.0 - optimal_error)
print("optimal parameters =", optimal_parameters)
