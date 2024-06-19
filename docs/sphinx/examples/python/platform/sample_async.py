# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Parallelize over the various kernels one would like to execute.

import cudaq

qubit_count = 2

# Set the simulation target.
cudaq.set_target("nvidia-mqpu")

# Kernel 1


@cudaq.kernel
def kernel_1(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)

    # 2-qubit GHZ state.
    h(qvector[0])
    for i in range(1, qubit_count):
        x.ctrl(qvector[0], qvector[i])

    # If we don't specify measurements, all qubits are measured in
    # the Z-basis by default.
    mz(qvector)


# Kernel 2


@cudaq.kernel
def kernel_2(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)

    # 2-qubit GHZ state.
    h(qvector[0])
    for i in range(1, qubit_count):
        x.ctrl(qvector[0], qvector[i])

    # If we don't specify measurements, all qubits are measured in
    # the Z-basis by default.
    mz(qvector)


if cudaq.num_available_gpus() > 1:
    # Asynchronous execution on multiple virtual QPUs, each simulated by an NVIDIA GPU.
    result_1 = cudaq.sample_async(kernel_1, qubit_count, shots_count=1000, qpu_id=0)
    result_2 = cudaq.sample_async(kernel_2, qubit_count, shots_count=1000, qpu_id=1)
else:
    # Schedule for execution on the same virtual QPU.
    result_1 = cudaq.sample_async(kernel_1, qubit_count, shots_count=1000, qpu_id=0)
    result_2 = cudaq.sample_async(kernel_2, qubit_count, shots_count=1000, qpu_id=0)

print(result_1.get())
print(result_2.get())
