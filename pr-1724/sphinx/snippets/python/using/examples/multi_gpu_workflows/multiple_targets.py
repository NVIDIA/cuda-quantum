# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin state]
import cudaq


def ghz_state(qubit_count, target):
    """A function that will generate a variable sized GHZ state (`qubit_count`)."""
    cudaq.set_target(target)

    kernel = cudaq.make_kernel()

    qubits = kernel.qalloc(qubit_count)

    kernel.h(qubits[0])

    for i in range(1, qubit_count):
        kernel.cx(qubits[0], qubits[i])

    kernel.mz(qubits)

    result = cudaq.sample(kernel, shots_count=1000)

    return result


# [End state]

# [Begin CPU]
cpu_result = ghz_state(qubit_count=2, target="qpp-cpu")
cpu_result.dump()
# [End CPU]

# [Begin GPU]
gpu_result = ghz_state(qubit_count=25, target="nvidia")
gpu_result.dump()
# [End GPU]
