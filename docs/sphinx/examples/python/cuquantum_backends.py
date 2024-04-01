# This example is meant to demonstrate the cuQuantum
# GPU-accelerated backends and their ability to easily handle
# a larger number of qubits compared the CPU-only backend.
#
# This will take a noticeably longer time to execute on
# CPU-only backends.

import cudaq

qubit_count = 5
# We can set a larger `qubit_count` if running on a GPU backend.
# qubit_count = 28


@cudaq.kernel
def kernel(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)
    h(qvector)
    for qubit in range(qubit_count - 1):
        x.ctrl(qvector[qubit], qvector[qubit + 1])
    mz(qvector)


result = cudaq.sample(kernel, qubit_count, shots_count=100)

if (not cudaq.mpi.is_initialized()) or (cudaq.mpi.rank() == 0):
    print(result)
