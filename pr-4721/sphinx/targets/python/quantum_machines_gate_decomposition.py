import cudaq
import math

# Test gate decomposition on the quantum machines target.
# The default executor is mock, use executor name to run on another backend (real or simulator).
# Configure the address of the QOperator server in the `url` argument, and set the `api_key`.
cudaq.set_target("quantum_machines",
                 url="http://host.docker.internal:8080",
                 api_key="1234567890",
                 executor="mock")

qubit_count = 5


@cudaq.kernel
def simplest():
    qvector = cudaq.qvector(qubit_count)

    reset(qvector)
    # `t.adj(qvector[0])` - not yet supported as `tdg` gate is not supported yet
    x.ctrl(qvector[0], qvector[1])
    h.ctrl(qvector[1], qvector[2])
    rz.adj(math.pi, qvector[3])
    cr1(math.pi, qvector[3], qvector[4])


cudaq.sample(simplest, shots_count=1000)
