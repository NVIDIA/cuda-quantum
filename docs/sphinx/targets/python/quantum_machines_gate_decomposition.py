import cudaq
import math


# simplest example of quantum machines target run:
# select a number of qubits (make sure you have this number of qubits available on your computer)
# and run a Hadamard on each of them.
# The default executor is mock, use executor name to run on another backend (real or simulator).
# Configure the address of the QOperator server in the url argument, and set the api_key
cudaq.set_target("quantum_machines",
                 url="http://host.docker.internal:8080",
                 api_key="1234567890",
                 executor="mock")

qubit_count = 5


@cudaq.kernel
def simplest():
    qvector = cudaq.qvector(qubit_count)
    for i in range(qubit_count):
        reset(qvector[i])

    # t.adj(qvector[0]) - not yet supported as tdg gate is not supported yet
    x.ctrl(qvector[0], qvector[1])
    h.ctrl(qvector[1], qvector[2])
    rz.adj(math.pi, qvector[3])
    cr1(math.pi, qvector[3], qvector[4])



cudaq.sample(simplest, shots_count=1000)
