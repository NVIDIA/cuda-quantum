import cudaq
from cudaq.operators import RydbergHamiltonian, ScalarOperator
from cudaq.dynamics import Schedule
import numpy as np

# This example illustrates how to use QuEra's Aquila device on Braket with CUDA-Q.
# It is a CUDA-Q implementation of the getting started materials for Braket available here:
# https://github.com/amazon-braket/amazon-braket-examples/blob/main/examples/analog_hamiltonian_simulation/01_Introduction_to_Aquila.ipynb

cudaq.set_target("quera")

separation = 5e-6
block_separation = 15e-6
k_max = 5
m_max = 5

register = []
for k in range(k_max):
    for m in range(m_max):
        register.append((block_separation * m,
                         block_separation * k + separation / np.sqrt(3)))
        register.append((block_separation * m - separation / 2,
                         block_separation * k - separation / (2 * np.sqrt(3))))
        register.append((block_separation * m + separation / 2,
                         block_separation * k - separation / (2 * np.sqrt(3))))

omega_const = 1.5e7  # rad / s
time_ramp = 5e-8  # s
time_plateau = 7.091995761561453e-08  # s
time_max = 2 * time_ramp + time_plateau  # s

# Schedule of time steps.
steps = [0, time_ramp, time_ramp + time_plateau, time_max]
schedule = Schedule(steps, ["t"])


def trapezoidal_signal(t):
    slope = omega_const / time_ramp
    y_intercept = slope * time_max
    if 0 < t.real < time_ramp + time_plateau:
        return slope * t
    if time_ramp < t.real < time_max:
        return omega_const
    if time_ramp + time_plateau < t.real < time_max:
        return (-slope * t) + y_intercept
    return 0.0


omega = ScalarOperator(lambda t: trapezoidal_signal(t))
phi = ScalarOperator.const(0.0)
delta = ScalarOperator.const(0.0)

evolution_result = cudaq.evolve(RydbergHamiltonian(atom_sites=register,
                                                   amplitude=omega,
                                                   phase=phi,
                                                   delta_global=delta),
                                schedule=schedule)
evolution_result.dump()
