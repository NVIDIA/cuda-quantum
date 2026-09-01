import cudaq
import math

# The default executor is mock, use executor name to run on another backend (real or simulator).
# Configure the address of the QOperator server in the `url` argument, and set the `api_key`.
cudaq.set_target(
    "quantum_machines",
    url="http://host.docker.internal:8000",
    #url="http://172.16.32.154:8000",
    #api_key="1234567890",
    #qubit_mapping_mode="backend",
    #qubit_mapping_mode="local_file",
    qubit_mapping_mode="local_get_latest",
    executor="iqcc")

qubit_count = 5


# Maximally entangled state between 5 qubits
@cudaq.kernel
def all_h():
    qvector = cudaq.qvector(qubit_count)

    #reset(`qvector`)
    for i in range(5):
        h(qvector[i])
    x.ctrl(qvector[0], qvector[1])
    x.ctrl(qvector[2], qvector[3])


# Define the Hamiltonian (Observable)
#hamiltonian = 5.0 * cudaq.spin.z(0)

# 2. Observe the observable (Expectation value)
#result = cudaq.observe(all_h, hamiltonian)
#print(f"Expectation Value: {result.expectation()}")

# Submit synchronously
cudaq.sample(all_h, shots_count=100).dump()
