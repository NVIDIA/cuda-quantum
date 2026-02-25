import cudaq
import math

# The default executor is mock, use executor name to run on another backend (real or simulator).
# Configure the address of the QOperator server in the `url` argument, and set the `api_key`.
cudaq.set_target("quantum_machines",
                 url="http://host.docker.internal:8080",
                 api_key="1234567890",
                 executor="mock")

qubit_count = 5


# Maximally entangled state between 5 qubits
@cudaq.kernel
def all_h():
    qvector = cudaq.qvector(qubit_count)

    for i in range(qubit_count - 1):
        h(qvector[i])

    s(qvector[0])
    r1(math.pi / 2, qvector[1])
    mz(qvector)


# Submit synchronously
cudaq.sample(all_h).dump()
