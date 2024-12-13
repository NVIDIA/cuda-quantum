# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]
import sys
import cudaq

print(f"Running on target {cudaq.get_target().name}")
qubit_count = int(sys.argv[1]) if 1 < len(sys.argv) else 2


@cudaq.kernel
def kernel():
    qubits = cudaq.qvector(qubit_count)
    h(qubits[0])
    for i in range(1, qubit_count):
        x.ctrl(qubits[0], qubits[i])
    mz(qubits)


result = cudaq.sample(kernel)
print(result)  # Example: { 11:500 00:500 }
# [End Documentation]
