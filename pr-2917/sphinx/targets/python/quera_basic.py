import cudaq
from cudaq.operators import RydbergHamiltonian, ScalarOperator
from cudaq.dynamics import Schedule
import numpy as np

## NOTE: QuEra Aquila system is available via Amazon Braket.
# Credentials must be set before running this program.
# Amazon Braket costs apply.

# This example illustrates how to use QuEra's Aquila device on Braket with CUDA-Q.
# It is a CUDA-Q implementation of the getting started materials for Braket available here:
# https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html

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
omega = ScalarOperator(lambda t: omega_max
                       if time_ramp < t.real < time_max else 0.0)
# Global phase at each step
phi = ScalarOperator.const(0.0)
# Global detuning at each step
delta = ScalarOperator(lambda t: delta_end
                       if time_ramp < t.real < time_max else delta_start)

async_result = cudaq.evolve_async(RydbergHamiltonian(atom_sites=register,
                                                     amplitude=omega,
                                                     phase=phi,
                                                     delta_global=delta),
                                  schedule=schedule,
                                  shots_count=10).get()
async_result.dump()

## Sample result
# ```
# {
#   __global__ : { 12121222:1 21202221:1 21212121:2 21212122:1 21221212:1 21221221:2 22121221:1 22221221:1 }
#    post_sequence : { 01010111:1 10101010:2 10101011:1 10101110:1 10110101:1 10110110:2 11010110:1 11110110:1 }
#    pre_sequence : { 11101111:1 11111111:9 }
# }
# ```

## Interpreting result
# `pre_sequence` has the measurement bits, one for each atomic site, before the
# quantum evolution is run. The count is aggregated across shots. The value is
# 0 if site is empty, 1 if site is filled.
# `post_sequence` has the measurement bits, one for each atomic site, at the
# end of the quantum evolution. The count is aggregated across shots. The value
# is 0 if atom is in Rydberg state or site is empty, 1 if atom is in ground
# state.
# `__global__` has the aggregate of the state counts from all the successful
# shots. The value is 0 if site is empty, 1 if atom is in Rydberg state (up
# state spin) and 2 if atom is in ground state (down state spin).
