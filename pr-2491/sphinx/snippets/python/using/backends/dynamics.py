# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Made up values - not sure what values are reasonable here.
omega_z = 6.5
omega_x = 4.0
omega_d = 0.5

#[Begin Transmon]
import numpy as np
from cudaq.operator import *

# Qubit Hamiltonian
hamiltonian = 0.5 * omega_z * spin.z(0)
# Add modulated driving term to the Hamiltonian
hamiltonian += omega_x * ScalarOperator(lambda t: np.cos(omega_d * t)) * spin.x(
    0)
#[End Transmon]

# Made up values - not sure what values are reasonable here.
t_final = 1.0
n_steps = 100

#[Begin Evolve]
import cudaq
import cupy as cp

# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Dimensions of sub-systems: a single two-level system.
dimensions = {0: 2}

# Initial state of the system (ground state).
rho0 = cudaq.State.from_data(
    cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

# Schedule of time steps.
steps = np.linspace(0, t_final, n_steps)
schedule = Schedule(steps, ["t"])

# Run the simulation.
evolution_result = evolve(hamiltonian,
                          dimensions,
                          schedule,
                          rho0,
                          observables=[spin.x(0),
                                       spin.y(0),
                                       spin.z(0)],
                          collapse_operators=[],
                          store_intermediate_results=True)
#[End Evolve]

#[Begin Plot]
get_result = lambda idx, res: [
    exp_vals[idx].expectation() for exp_vals in res.expectation_values()
]

import matplotlib.pyplot as plt

plt.plot(steps, get_result(0, evolution_result))
plt.plot(steps, get_result(1, evolution_result))
plt.plot(steps, get_result(2, evolution_result))
plt.ylabel("Expectation value")
plt.xlabel("Time")
plt.legend(("Sigma-X", "Sigma-Y", "Sigma-Z"))
#[End Plot]

# Made up values - not sure what values are reasonable here.
omega_c = 6 * np.pi
omega_a = 4 * np.pi
Omega = 0.5

#[Begin Jaynes-Cummings]
hamiltonian = omega_c * operators.create(1) * operators.annihilate(1) \
                + (omega_a / 2) * spin.z(0) \
                + (Omega / 2) * (operators.annihilate(1) * spin.plus(0) + operators.create(1) * spin.minus(0))
#[End Jaynes-Cummings]

# Made up values - not sure what values are reasonable here.
omega = np.pi

#[Begin Hamiltonian]
# Define the static (drift) and control terms
H0 = spin.z(0)
H1 = spin.x(0)
H = H0 + ScalarOperator(lambda t: np.cos(omega * t)) * H1
#[End Hamiltonian]

#[Begin DefineOp]
import numpy
import scipy
from cudaq.operator import *
from numpy.typing import NDArray


def displacement_matrix(
        dimension: int,
        displacement: NumericType) -> NDArray[numpy.complexfloating]:
    """
    Returns the displacement operator matrix.
    Args:
        displacement: Amplitude of the displacement operator.
            See also https://en.wikipedia.org/wiki/Displacement_operator.
    """
    displacement = complex(displacement)
    term1 = displacement * operators.create(0).to_matrix({0: dimension})
    term2 = numpy.conjugate(displacement) * operators.annihilate(0).to_matrix(
        {0: dimension})
    return scipy.linalg.expm(term1 - term2)


# The second argument here indicates the the defined operator
# acts on a single degree of freedom, which can have any dimension.
# An argument [2], for example, would indicate that it can only
# act on a single degree of freedom with dimension two.
ElementaryOperator.define("displace", [0], displacement_matrix)


def displacement(degree: int) -> ElementaryOperator:
    """
    Instantiates a displacement operator acting on the given degree of freedom.
    """
    return ElementaryOperator("displace", [degree])


#[End DefineOp]

#[Begin Schedule1]
import cudaq

# Define a system consisting of a single degree of freedom (0) with dimension 3.
system_dimensions = {0: 3}
system_operator = displacement(0)

# Define the time dependency of the system operator as a schedule that linearly
# increases the displacement parameter from 0 to 1.
time_dependence = Schedule(numpy.linspace(0, 1, 100), ['displacement'])
initial_state = cudaq.State.from_data(
    numpy.ones(3, dtype=numpy.complex128) / numpy.sqrt(3))

# Simulate the evolution of the system under this time dependent operator.
cudaq.evolve(system_operator, system_dimensions, time_dependence, initial_state)
#[End Schedule1]

#[Begin Schedule2]
system_operator = displacement(0) + operators.squeeze(0)


# Define a schedule such that displacement amplitude increases linearly in time
# but the squeezing amplitude decreases, that is follows the inverse schedule.
def parameter_values(time_steps):

    def compute_value(param_name, step_idx):
        match param_name:
            case 'displacement':
                return time_steps[int(step_idx)]
            case 'squeezing':
                return time_steps[-int(step_idx + 1)]
            case _:
                raise ValueError(f"value for parameter {param_name} undefined")

    return Schedule(range(len(time_steps)), system_operator.parameters.keys(),
                    compute_value)


time_dependence = parameter_values(numpy.linspace(0, 1, 100))
cudaq.evolve(system_operator, system_dimensions, time_dependence, initial_state)
#[End Schedule2]
