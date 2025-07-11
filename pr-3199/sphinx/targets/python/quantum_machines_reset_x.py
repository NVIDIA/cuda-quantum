import cudaq

# Test resetting and applying an X gate on the quantum machines target.
# The default executor is mock, use executor name to run on another backend (real or simulator).
# Configure the address of the QOperator server in the `url` argument, and set the `api_key`.
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
