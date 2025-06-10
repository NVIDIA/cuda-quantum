import cudaq


# simplest example of quantum machines target run:
# select a number of qubits (make sure you have this number of qubits available on your computer)
# and run a Hadamard on each of them.
# The default executor is mock, use executor name to run on another backend (real or simulator).
# Configure the address of the QOperator server in the URL argument, and set the API key.
cudaq.set_target("quantum_machines",
                url="http://host.docker.internal:8080",
                api_key="1234567890", executor="mock")

qubit_count = 5


@cudaq.kernel
def gates():
    qvector = cudaq.qvector(qubit_count)
    for i in range(qubit_count):
        reset(qvector[i])

    # for i in range(qubit_count):
    #     x(qvector[i])

    x(qvector[0])
    y(qvector[0])
    z(qvector[0])
    rx(math.pi/2, qvector[1])
    ry(math.pi/2, qvector[1])
    rz(math.pi/2, qvector[1])
    s(qvector[2])
    t(qvector[3])
    r1(math.pi/2, qvector[4])


cudaq.sample(gates, shots_count=1000)
