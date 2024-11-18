import cudaq
from cudaq.operator import *
import numpy as np

## NOTE: QuEra Aquila system is available via AWS Braket.
# AWS credentials must be set before running this program.
# AWS Braket costs apply.

cudaq.set_target("quera")

# Define the 2-dimensional atom arrangement
a = 5.7e-6
register = []
register.append(tuple(np.array([0.5, 0.5 + 1 / np.sqrt(2)]) * a))
register.append(tuple(np.array([0.5 + 1 / np.sqrt(2), 0.5]) * a))
register.append(tuple(np.array([0.5 + 1 / np.sqrt(2), -0.5]) * a))
register.append(tuple(np.array([0.5, -0.5 - 1 / np.sqrt(2)]) * a))
register.append(tuple(np.array([-0.5, -0.5 - 1 / np.sqrt(2)]) * a))
register.append(tuple(np.array([-0.5 - 1 / np.sqrt(2), -0.5]) * a))
register.append(tuple(np.array([-0.5 - 1 / np.sqrt(2), 0.5]) * a))
register.append(tuple(np.array([-0.5, 0.5 + 1 / np.sqrt(2)]) * a))

time_max = 4e-6  # seconds
time_ramp = 1e-7  # seconds
omega_max = 6300000.0  # rad / sec
delta_start = -5 * omega_max
delta_end = 5 * omega_max

# Times for the piece-wise linear waveforms
steps = [0.0, time_ramp, time_max - time_ramp, time_max]
schedule = Schedule(steps, ["t"])
# Rabi frequencies at each step
omega = ScalarOperator(lambda t: omega_max if time_ramp < t < time_max else 0.0)
# Global phase at each step
phi = ScalarOperator.const(0.0)
# Global detuning at each step
delta = ScalarOperator(lambda t: delta_end
                       if time_ramp < t < time_max else delta_start)

async_result = evolve_async(RydbergHamiltonian(atom_sites=register,
                                               amplitude=omega,
                                               phase=phi,
                                               delta_global=delta),
                            schedule=schedule,
                            shots_count=10).get()
async_result.dump()
