import cudaq
from cudaq.operators import RydbergHamiltonian, ScalarOperator
from cudaq.dynamics import Schedule

# This example illustrates how to use Pasqal's EMU_MPS emulator over Pasqal's cloud via CUDA-Q.
#
# To obtain the authentication token for the cloud  we recommend logging in with
# Pasqal's Python SDK. See our documentation https://docs.pasqal.com/cloud/ for more.
#
# Contact Pasqal at help@pasqal.com or through https://community.pasqal.com for assistance.
#
# Visit the documentation portal, https://docs.pasqal.com/, to find further
# documentation on Pasqal's devices, emulators and the cloud platform.
#
# For more details on the EMU_MPS emulator see the documentation of the open-source
# package: https://pasqal-io.github.io/emulators/latest/emu_mps/.
from pasqal_cloud import SDK
import os

# We recommend leaving the password empty in an interactive session as you will be
# prompted to enter it securely via the command line interface.
sdk = SDK(
    username=os.environ.get("PASQAL_USERNAME"),
    password=os.environ.get("PASQAL_PASSWORD", None),
)

os.environ["PASQAL_AUTH_TOKEN"] = str(sdk.user_token())

# It is also mandatory to specify the project against which the execution will be billed.
# Uncomment this line to set it from Python, or export it as an environment variable
# prior to execution. You can find your projects here: https://portal.pasqal.cloud/projects.
# ```
# os.environ['PASQAL_PROJECT_ID'] = 'your project id'
# ```

# Set the target including specifying optional arguments like target machine
cudaq.set_target("pasqal",
                 machine=os.environ.get("PASQAL_MACHINE_TARGET", "EMU_MPS"))

# ```
## To target QPU set FRESNEL as the machine, see our cloud portal for latest machine names
# cudaq.set_target("pasqal", machine="FRESNEL")
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
                                  shots_count=100).get()
async_result.dump()

## Sample result
# ```
# {'001': 16, '010': 23, '100': 19, '000': 42}
# ```
