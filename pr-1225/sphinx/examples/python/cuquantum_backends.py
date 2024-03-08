# This example is meant to demonstrate the cuQuantum
# GPU-accelerated backends and their ability to easily handle
# a larger number of qubits compared the CPU-only backend.
#
# This will take a noticeably longer time to execute on
# CPU-only backends.

import cudaq

qubit_count = 28

kernel = cudaq.make_kernel()
qvector = kernel.qalloc(qubit_count)
kernel.h(qvector[0])
for qubit in range(qubit_count - 1):
    kernel.cx(qvector[qubit], qvector[qubit + 1])
kernel.mz(qvector)

result = cudaq.sample(kernel, shots_count=100)

if (not cudaq.mpi.is_initialized()) or (cudaq.mpi.rank() == 0):
    print(result)
