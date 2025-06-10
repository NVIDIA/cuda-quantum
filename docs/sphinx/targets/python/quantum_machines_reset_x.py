import cudaq

# simplest example of quantum machines target run:
# select a number of qubits (make sure you have this number of qubits available on your computer)
# and run a Hadamard on each of them.
# The default executor is mock, use executor name to run on another backend (real or simulator).
# Configure the address of the QOperator server in the URL argument, and set the API key.
cudaq.set_target("quantum_machines",
                 url="http://host.docker.internal:8080",
                 api_key="1234567890",
                 executor="mock")

qubit_count = 4


@cudaq.kernel
def reset_and_x():
    qvector = cudaq.qvector(qubit_count)
    reset(qvector)

    for i in range(qubit_count):
        x(qvector[i])


cudaq.sample(reset_and_x, shots_count=10)
