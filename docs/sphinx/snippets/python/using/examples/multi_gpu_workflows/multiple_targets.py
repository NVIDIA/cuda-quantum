# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin state]
import cudaq


@cudaq.kernel
def ghz_state(qubit_count: int):
    qubits = cudaq.qvector(qubit_count)
    h(qubits[0])
    for i in range(1, qubit_count):
        cx(qubits[0], qubits[i])
    mz(qubits)


def sample_ghz_state(qubit_count, target):
    """A function that will sample a variable sized GHZ state."""
    cudaq.set_target(target)
    result = cudaq.sample(ghz_state, qubit_count, shots_count=1000)
    return result


# [End state]

# [Begin CPU]
cpu_result = sample_ghz_state(qubit_count=2, target="qpp-cpu")
cpu_result.dump()
# [End CPU]

# [Begin GPU]
if cudaq.num_available_gpus() > 0:
    gpu_result = sample_ghz_state(qubit_count=25, target="nvidia")
    gpu_result.dump()
# [End GPU]
