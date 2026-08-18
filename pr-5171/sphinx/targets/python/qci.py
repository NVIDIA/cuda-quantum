import cudaq

# Make sure to export or otherwise present your user token via the environment,
# e.g., using export:
# ```
# export QCI_AUTH_TOKEN="your token here"
# ```
#
# The example will run on QCI's AquSim simulator by default.

cudaq.set_target("qci")


@cudaq.kernel
def teleportation():

    # Initialize a three qubit quantum circuit
    qubits = cudaq.qvector(3)

    # Random quantum state on qubit 0.
    rx(3.14, qubits[0])
    ry(2.71, qubits[0])
    rz(6.62, qubits[0])

    # Create a maximally entangled state on qubits 1 and 2.
    h(qubits[1])
    cx(qubits[1], qubits[2])

    cx(qubits[0], qubits[1])

    h(qubits[0])
    m1 = mz(qubits[0])
    m2 = mz(qubits[1])

    if m1 == 1:
        z(qubits[2])

    if m2 == 1:
        x(qubits[2])

    mz(qubits)


print(cudaq.sample(teleportation))
