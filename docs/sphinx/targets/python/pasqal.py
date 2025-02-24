import cudaq
from cudaq.operator import *

# This example illustrates how to use Pasqal's Fresnel device over Pasqal's cloud via CUDA-Q.

# To login via Pasqal's SDK uncomment these lines
# We recommend filling in password via an environment variable
# or leave it empty in an interactive session as you will be
# promted to enter the password securely via the command line.
# from pasqal_cloud import SDK
# import os
# username = <Your email on Pasqal's cloud platform>
# password = <The corresponding password or None to enter from CMD line>
# project_id = <The project_id to bill against>
# sdk = SDK(
#     username=username,
#     password=password,
#     project_id=project_id,
# )
# token = sdk._client.authenticator.token_provider.get_token()
# os.environ["PASQAL_AUTH_TOKEN"] = str(token)
# os.environ["PASQAL_PROJECT_ID"] = project_id


cudaq.set_target("pasqal")

# Define the 2-dimensional atom arrangement
a = 5e-6
register = [(a,0), (2*a, 0), (3*a, 0)]
time_ramp = 0.000002
time_max  = 0.000003
# Times for the piece-wise linear waveforms
steps = [0.0, time_ramp, time_max - time_ramp, time_max]
schedule = Schedule(steps, ["t"])
# Rabi frequencies at each step
omega_max=100000
delta_end=100000
delta_start=0.0
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
#    counter : {'001': 16, '010': 23, '100': 19, '000': 42}
# }
# ```
