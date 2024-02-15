import cudaq
from cudaq import spin
import math

# This example assumes the NVCF API key and Function Id have been set in the `~/.nvcf_config` file/environment variables.
# If not, you can set the API Key and Function ID environment variables in the Python script with:
# ```
# os.environ["NVCF_API_KEY"] = "<YOUR NVCF API KEY>"`
# os.environ["NVCF_FUNCTION_ID"] = "<YOUR NVCF FUNCTION ID>"
# ```
# Alternatively, the `api_key` and `function_id` values can be passed to the target directly,
# ```
# cudaq.set_target("nvcf",
#                 nqpus=3,
#                 api_key="<YOUR NVCF API KEY>"
#                 function_id="<YOUR NVCF FUNCTION ID>")
# ```
cudaq.set_target("nvcf", nqpus=3)

print("Number of QPUs:", cudaq.get_target().num_qpus())
# Note: depending on the user's account, there might be different
# number of NVCF worker instances available. Hence, although we're making
# concurrent job submissions across multiple QPUs, the speedup would be
# determined by the number of NVCF worker instances.
# Create the parameterized ansatz
@cudaq.kernel(jit=True)
def ansatz(theta: float):
    qubits = cudaq.qvector(2)
    x(qubits[0])
    ry(theta, qubits[1])
    x.ctrl(qubits[1], qubits[0]) 

# Define its spin Hamiltonian.
hamiltonian = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
               2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
               6.125 * spin.z(1))


def opt_gradient(parameter_vector):
    # Evaluate energy and gradient on different remote QPUs
    # (i.e., concurrent job submissions to NVCF)
    energy_future = cudaq.observe_async(ansatz,
                                        hamiltonian,
                                        parameter_vector[0],
                                        qpu_id=0)
    plus_future = cudaq.observe_async(ansatz,
                                      hamiltonian,
                                      parameter_vector[0] + 0.5 * math.pi,
                                      qpu_id=1)
    minus_future = cudaq.observe_async(ansatz,
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
