import cudaq
from cudaq.operator import *

# This example illustrates how to use Pasqal's EMU_MPS emulator over Pasqal's cloud via CUDA-Q.

# To obtain the authentication token we recommend logging in
# by using Pasqal's Python SDK.
# Fill in the  username and password via the environment variables.
# The password can be left empty in an interactive session to be
# prompted to enter the it securely via the command line interface.
#
# Contact Pasqal at help@pasqal.com or through https://community.pasqal.com for assistance.
#
# See our general docs https://docs.pasqal.com/cloud/set-up/
# to see how to get this setup, or the CUDA-Q documentation at
# https://nvidia.github.io/cuda-quantum/latest/using/backends/hardware/neutralatom.html#pasqal
from pasqal_cloud import SDK
import os

sdk = SDK(
    username=os.environ.get("PASQAL_USERNAME"),
    password=os.environ.get("PASQAL_PASSWORD", None),
)
token = sdk._client.authenticator.token_provider.get_token()

os.environ["PASQAL_AUTH_TOKEN"] = str(token)

cudaq.set_target("pasqal",
                 machine=os.environ.get("PASQAL_MACHINE_TARGET", "EMU_MPS"))

# ```
# cudaq.set_target("pasqal", machine="FRESNEL") ## To target QPU
# ```

# Define the 2-dimensional atom arrangement
a = 5e-6
register = [(a, 0), (2 * a, 0), (3 * a, 0)]
time_ramp = 0.000001
time_max = 0.000003
# Times for the piece-wise linear waveforms
steps = [0.0, time_ramp, time_max - time_ramp, time_max]
schedule = Schedule(steps, ["t"])
# Rabi frequencies at each step
omega_max = 1000000
delta_end = 1000000
delta_start = 0.0
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
                            shots_count=100).get()
async_result.dump()

## Sample result
# ```
# {'001': 16, '010': 23, '100': 19, '000': 42}
# ```
