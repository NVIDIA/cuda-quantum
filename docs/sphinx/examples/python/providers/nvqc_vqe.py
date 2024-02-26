import cudaq
from cudaq import spin
import math

# This example assumes the NVQC API key has been set in the `~/.nvqc_config` file/environment variables.
# If not, you can set the API Key environment variable in the Python script with:
# ```
# os.environ["NVQC_API_KEY"] = "<YOUR NVQC API KEY>"`
# ```
# Alternatively, the `api_key` value can be passed to the target directly,
# ```
# cudaq.set_target("nvqc",
#                 nqpus=3,
#                 api_key="<YOUR NVQC API KEY>")
# ```
cudaq.set_target("nvqc", nqpus=3)

print("Number of QPUs:", cudaq.get_target().num_qpus())
# Note: depending on the user's account, there might be different
# number of NVQC worker instances available. Hence, although we're making
# concurrent job submissions across multiple QPUs, the speedup would be
# determined by the number of NVQC worker instances.
# Create the parameterized ansatz
kernel, theta = cudaq.make_kernel(float)
qreg = kernel.qalloc(2)
kernel.x(qreg[0])
kernel.ry(theta, qreg[1])
kernel.cx(qreg[1], qreg[0])

# Define its spin Hamiltonian.
hamiltonian = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
               2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
               6.125 * spin.z(1))


def opt_gradient(parameter_vector):
    # Evaluate energy and gradient on different remote QPUs
    # (i.e., concurrent job submissions to NVQC)
    energy_future = cudaq.observe_async(kernel,
                                        hamiltonian,
                                        parameter_vector[0],
                                        qpu_id=0)
    plus_future = cudaq.observe_async(kernel,
                                      hamiltonian,
                                      parameter_vector[0] + 0.5 * math.pi,
                                      qpu_id=1)
    minus_future = cudaq.observe_async(kernel,
                                       hamiltonian,
                                       parameter_vector[0] - 0.5 * math.pi,
                                       qpu_id=2)
    return (energy_future.get().expectation(), [
        (plus_future.get().expectation() - minus_future.get().expectation()) /
        2.0
    ])


optimizer = cudaq.optimizers.LBFGS()
optimal_value, optimal_parameters = optimizer.optimize(1, opt_gradient)
print("Ground state energy =", optimal_value)
print("Optimal parameters =", optimal_parameters)
