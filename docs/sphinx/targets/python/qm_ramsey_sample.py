import cudaq
import math

# The default executor is mock, use executor name to run on another backend (real or simulator).
# Configure the address of the QOperator server in the `url` argument, and set the `api_key`.
cudaq.set_target("quantum_machines",
                 url="http://host.docker.internal:8000",
                 #url="http://172.16.32.154:8000",
                 #api_key="1234567890",
                 qubit_mapping_mode="backend",
                 #qubit_mapping_mode="local_file",
                 #qubit_mapping_mode="local_get_latest",
                 executor="sim")

qubit_count = 5


# Maximally entangled state between 5 qubits
@cudaq.kernel
def ramsey_single(wait_duration: float):
    qubit = cudaq.qubit()

    reset(qubit)
    rx(np.pi / 2, qubit)
    wait(wait_duration, qubit)
    rx(np.pi / 2, qubit)
    

wait_duration_to_mz_1 = []
sweep = [x * 1. for x in range(1, 10)]
for wait_duration in sweep:
    result = cudaq.sample(ramsey_single, [wait_duration], shots_count=2)
    #wait_duration_to_mz_1.append((wait_duration, result.counts().get('1', 0) / 100.0))
    wait_duration_to_mz_1.append((wait_duration, result))

print(wait_duration_to_mz_1)


