import cudaq
from cudaq import spin
import math

# This example assumes the NVQC API key has been set in the `NVQC_API_KEY` environment variable.
# If not, you can set the API Key environment variable in the Python script with:
# ```
# os.environ["NVQC_API_KEY"] = "<YOUR NVQC API KEY>"`
# ```

cudaq.set_target("nvqc", nqpus=3)

print("Number of QPUs:", cudaq.get_target().num_qpus())


# Note: depending on the user's account, there might be different
# number of NVQC worker instances available. Hence, although we're making
# concurrent job submissions across multiple QPUs, the speedup would be
# determined by the number of NVQC worker instances.
# Create the parameterized ansatz
@cudaq.kernel
def ansatz(theta: float):
    qvector = cudaq.qvector(2)
    x(qvector[0])
    ry(theta, qvector[1])
    x.ctrl(qvector[1], qvector[0])


# Define its spin Hamiltonian.
hamiltonian = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
               2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
               6.125 * spin.z(1))


def opt_gradient(parameter_vector):
    # Evaluate energy and gradient on different remote QPUs
    # (i.e., concurrent job submissions to NVQC)
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
