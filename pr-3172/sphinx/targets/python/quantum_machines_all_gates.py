import cudaq

# Test all gates on the quantum machines target.
# The default executor is mock, use executor name to run on another backend (real or simulator).
# Configure the address of the QOperator server in the `url` argument, and set the `api_key`.
cudaq.set_target("quantum_machines",
                 url="http://host.docker.internal:8080",
                 api_key="1234567890",
                 executor="mock")

qubit_count = 5


@cudaq.kernel
def gates():
    qvector = cudaq.qvector(qubit_count)
    reset(qvector)

    x(qvector[0])
    y(qvector[0])
    z(qvector[0])
    rx(math.pi / 2, qvector[1])
    ry(math.pi / 2, qvector[1])
    rz(math.pi / 2, qvector[1])
    s(qvector[2])
    t(qvector[3])
    r1(math.pi / 2, qvector[4])


cudaq.sample(gates, shots_count=10000)
