import cudaq
from cudaq import spin

# We begin by defining the spin Hamiltonian for the system that we are working
# with. This is achieved through the use of `cudaq.SpinOperator`'s, which allow
# for the convenient creation of complex Hamiltonians out of Pauli spin operators.
hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
    0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

# Next, using the `cudaq.Kernel`, we define the variational quantum circuit
# that we'd like to use as an ansatz.
# Create a kernel that takes a list of floats as a function argument.
kernel, thetas = cudaq.make_kernel(list)
# Allocate 2 qubits.
qubits = kernel.qalloc(2)
kernel.x(qubits[0])
# Apply an `ry` gate that is parameterized by the first
# `QuakeValue` entry of our list, `thetas`.
kernel.ry(thetas[0], qubits[1])
kernel.cx(qubits[1], qubits[0])
# Note: the kernel must not contain measurement instructions.

# The last thing we need is to pick an optimizer from the suite of `cudaq.optimizers`.
# We can optionally tune this optimizer through its initial parameters, iterations,
# optimization bounds, etc. before passing it to `cudaq.vqe`.
optimizer = cudaq.optimizers.COBYLA()
# optimizer.max_iterations = ...
# optimizer...

# Finally, we can pass all of that into `cudaq.vqe` and it will automatically run our
# optimization loop and return a tuple of the minimized eigenvalue of our `spin_operator`
# and the list of optimal variational parameters.
energy, parameter = cudaq.vqe(
    kernel=kernel,
    spin_operator=hamiltonian,
    optimizer=optimizer,
    # list of parameters has length of 1:
    parameter_count=1)

print(f"\nminimized <H> = {round(energy,16)}")
print(f"optimal theta = {round(parameter[0],16)}")
