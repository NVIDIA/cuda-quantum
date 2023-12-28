import cudaq
import random

cudaq.enable_jit()


def random_bitstring(length: int):
    bitstring = []
    for _ in range(length):
        bitstring.append(random.randint(0, 1))
    return bitstring


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

# Generate a random, hidden bitstring for the oracle
# to encode. Note: we define the bitstring here so
# as to be able to return it for verification.
hidden_bitstring = random_bitstring(qubit_count)


@cudaq.kernel  # or can pass (jit=True)
def oracle(register: cudaq.qview, auxillary_qubit: cudaq.qubit,
           hidden_bitstring: list[int]):
    for index, bit in enumerate(hidden_bitstring):
        if bit == 1:
            # apply a `cx` gate with the current qubit as
            # the control and the auxillary qubit as the target.
            x.ctrl(register[index], auxillary_qubit)


@cudaq.kernel
def bernstein_vazirani(hidden_bitstring: list[int]):
    # Allocate the specified number of qubits - this
    # corresponds to the length of the hidden bitstring.
    qubits = cudaq.qvector(len(hidden_bitstring))
    # Allocate an extra auxillary qubit.
    auxillary_qubit = cudaq.qubit()

    # Prepare the auxillary qubit.
    h(auxillary_qubit)
    z(auxillary_qubit)

    # Place the rest of the register in a superposition state.
    h(qubits)

    # Query the oracle.
    oracle(qubits, auxillary_qubit, hidden_bitstring)

    # Apply another set of Hadamards to the register.
    h(qubits)

    # Apply measurement gates to just the `qubits`
    # (excludes the auxillary qubit).
    mz(qubits)


print(bernstein_vazirani)
result = cudaq.sample(bernstein_vazirani, hidden_bitstring)

print(f"encoded bitstring = {hidden_bitstring}")
print(f"measured state = {result.most_probable()}")
print(
    f"Were we successful? {''.join([str(i) for i in hidden_bitstring]) == result.most_probable()}"
)
