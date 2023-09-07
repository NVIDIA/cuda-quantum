import cudaq
from cudaq import spin

from typing import List, Tuple

# We will be optimizing over a custom objective function that takes a vector
# of parameters as input and returns either the cost as a single float,
# or in a tuple of (cost, gradient_vector) depending on the optimizer used.

# In this case, we will use the spin Hamiltonian and ansatz from `simple_vqe.py`
# and find the `thetas` that minimize the expectation value of the system.
hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
    0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

kernel, thetas = cudaq.make_kernel(list)
qubits = kernel.qalloc(2)
kernel.x(qubits[0])
kernel.ry(thetas[0], qubits[1])
kernel.cx(qubits[1], qubits[0])

# Define the optimizer that we'd like to use.
optimizer = cudaq.optimizers.Adam()

# Since we'll be using a gradient-based optimizer, we can leverage
# CUDA Quantum's gradient helper class to automatically compute the gradient
# vector for us. The use of this class for gradient calculations is
# purely optional and can be replaced with your own custom gradient
# routine.
gradient = cudaq.gradients.CentralDifference()


def objective_function(parameter_vector: List[float],
                       hamiltonian=hamiltonian,
                       gradient_strategy=gradient,
                       kernel=kernel) -> Tuple[float, List[float]]:
    """
    Note: the objective function may also take extra arguments, provided they
    are passed into the function as default arguments in python.
    """

    # Call `cudaq.observe` on the spin operator and ansatz at the
    # optimizer provided parameters. This will allow us to easily
    # extract the expectation value of the entire system in the
    # z-basis.

    # We define the call to `cudaq.observe` here as a lambda to
    # allow it to be passed into the gradient strategy as a
    # function. If you were using a gradient-free optimizer,
    # you could purely define `cost = cudaq.observe().expectation_z()`.
    get_result = lambda parameter_vector: cudaq.observe(
        kernel, hamiltonian, parameter_vector, shots_count=100).expectation_z()
    # `cudaq.observe` returns a `cudaq.ObserveResult` that holds the
    # counts dictionary and the `expectation_z`.
    cost = get_result(parameter_vector)
    print(f"<H> = {cost}")
    # Compute the gradient vector using `cudaq.gradients.STRATEGY.compute()`.
    gradient_vector = gradient_strategy.compute(parameter_vector, get_result,
                                                cost)

    # Return the (cost, gradient_vector) tuple.
    return cost, gradient_vector


cudaq.set_random_seed(13)  # make repeatable
energy, parameter = optimizer.optimize(dimensions=1,
                                       function=objective_function)

print(f"\nminimized <H> = {round(energy,16)}")
print(f"optimal theta = {round(parameter[0],16)}")
