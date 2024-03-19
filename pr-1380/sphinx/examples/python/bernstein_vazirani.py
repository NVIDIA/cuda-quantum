import cudaq
import random

from typing import List


def random_bits(length: int):
    bitset = []
    for _ in range(length):
        bitset.append(random.randint(0, 1))
    return bitset


# If you have a NVIDIA GPU you can use this example to see
# that the GPU-accelerated backends can easily handle a
# larger number of qubits compared the CPU-only backend.

# Depending on the available memory on your GPU, you can
# set the number of qubits to around 30 qubits, and un-comment
# the `cudaq.set_target(nvidia)` line.

# Note: Without setting the target to the `nvidia` backend,
# there will be a noticeable decrease in simulation performance.
# This is because the CPU-only backend has difficulty handling
# 30+ qubit simulations.

qubit_count = 5  # set to around 30 qubits for `nvidia` target
# ```
# cudaq.set_target("nvidia")
# ```

# Generate a random, hidden bitstring for the oracle
# to encode. Note: we define the bitstring here so
# as to be able to return it for verification.
hidden_bits = random_bits(qubit_count)


@cudaq.kernel
def oracle(register: cudaq.qview, auxillary_qubit: cudaq.qubit,
           hidden_bits: List[int]):
    for index, bit in enumerate(hidden_bits):
        if bit == 1:
            # apply a `cx` gate with the current qubit as
            # the control and the auxillary qubit as the target.
            x.ctrl(register[index], auxillary_qubit)


@cudaq.kernel
def bernstein_vazirani(hidden_bits: List[int]):
    # Allocate the specified number of qubits - this
    # corresponds to the length of the hidden bitstring.
    qubits = cudaq.qvector(len(hidden_bits))
    # Allocate an extra auxillary qubit.
    auxillary_qubit = cudaq.qubit()

    # Prepare the auxillary qubit.
    h(auxillary_qubit)
    z(auxillary_qubit)

    # Place the rest of the register in a superposition state.
    h(qubits)

    # Query the oracle.
    oracle(qubits, auxillary_qubit, hidden_bits)

    # Apply another set of Hadamards to the register.
    h(qubits)

    # Apply measurement gates to just the `qubits`
    # (excludes the auxillary qubit).
    mz(qubits)


print(cudaq.draw(bernstein_vazirani, hidden_bits))
result = cudaq.sample(bernstein_vazirani, hidden_bits)

print(f"encoded bitstring = {hidden_bits}")
print(f"measured state = {result.most_probable()}")
print(
    f"Were we successful? {''.join([str(i) for i in hidden_bits]) == result.most_probable()}"
)
