import cudaq
from cudaq.operator import *

# This example illustrates how to use Pasqal's Fresnel device over Pasqal's cloud via CUDA-Q.

cudaq.set_target("pasqal")

# Define the 2-dimensional atom arrangement
a = 5.7e-6
register = []

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

## Sample result
# ```
# {
#    counter : { 01010111:1 10101010:2 10101011:1 10101110:1 10110101:1 10110110:2 11010110:1 11110110:1 }
# }
# ```
