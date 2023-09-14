import cudaq
import random


def random_bitstring(length: int):
    bitstring = ""
    for bit in range(length):
        bitstring += str(random.randint(0, 1))
    return bitstring


def oracle(kernel: cudaq.Kernel, register: cudaq.QuakeValue,
           auxillary_qubit: cudaq.QuakeValue, hidden_bitstring: str):
    """
    The inner-product oracle for Bernstein Vazirani.
    """
    for index, bit in enumerate(hidden_bitstring):
        if bit == "0":
            # Apply identity operation to the qubit if it's
            # to be in the 0-state.
            # In this case, we do nothing.
            pass
        else:
            # Otherwise, apply a `cx` gate with the current qubit as
            # the control and the auxillary qubit as the target.
            kernel.cx(control=register[index], target=auxillary_qubit)


def bernstein_vazirani(qubit_count: int):
    """
    Returns a kernel implementing the Bernstein Vazirani algorithm
    for a random, hidden bitstring.
    """
    kernel = cudaq.make_kernel()
    # Allocate the specified number of qubits - this
    # corresponds to the length of the hidden bitstring.
    qubits = kernel.qalloc(qubit_count)
    # Allocate an extra auxillary qubit.
    auxillary_qubit = kernel.qalloc()

    # Prepare the auxillary qubit.
    kernel.h(auxillary_qubit)
    kernel.z(auxillary_qubit)

    # Place the rest of the register in a superposition state.
    kernel.h(qubits)

    # Generate a random, hidden bitstring for the oracle
    # to encode. Note: we define the bitstring here so
    # as to be able to return it for verification.
    hidden_bitstring = random_bitstring(qubit_count)

    # Query the oracle.
    oracle(kernel, qubits, auxillary_qubit, hidden_bitstring)

    # Apply another set of Hadamards to the register.
    kernel.h(qubits)

    # Apply measurement gates to just the `qubits`
    # (excludes the auxillary qubit).
    kernel.mz(qubits)
    return kernel, hidden_bitstring


# If you have a NVIDIA GPU you can use this example to see
# that the GPU-accelerated backends can easily handle a
# larger number of qubits compared the CPU-only backend.

# Depending on the available memory on your GPU, you can
# set the number of qubits to around 30 qubits, and un-comment
# the `cudaq.set_target(nvidia)` line.

# Note: Without setting the target to the `nvidia` backend,
# a 30 qubit simulation simply seems to hang; that is
# because it takes a long time for the CPU-only backend
# to handle this number of qubits!

qubit_count = 5  # set to around 30 qubits for `nvidia` target
# ```
# cudaq.set_target("nvidia")
# ```

kernel, hidden_bitstring = bernstein_vazirani(qubit_count)
result = cudaq.sample(kernel)

print(f"encoded bitstring = {hidden_bitstring}")
print(f"measured state = {result.most_probable()}")
print(f"Were we successful? {hidden_bitstring == result.most_probable()}")
