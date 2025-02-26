import cudaq
from cudaq.operator import *

# This example illustrates how to use Pasqal's EMU_MPS emulator over Pasqal's cloud via CUDA-Q.

# To obtain login we recommend using Pasqal's Python SDK
# We recommend filling in password via an environment variable
# or leave it empty in an interactive session as you will be
# prompted to enter the password securely via the command line.
# Contact Pasqal at help@pasqal.com or at https://community.pasqal.com for assistance.
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
                 machine=os.environ.get("PASQAL_MACHINE_TARGET", None))

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
                            shots_count=10).get()
async_result.dump()

# TODO: We don't have "counter" key in result right now just the dict
# check if we should in order to conform with standard
## Sample result
# ```
# {
#    counter : {'001': 16, '010': 23, '100': 19, '000': 42}
# }
# ```
