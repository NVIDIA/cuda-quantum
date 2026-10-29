# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]
import cudaq


# A GHZ-state kernel standing in for what a real project would provide.
@cudaq.kernel
def ghz(qubit_count: int):
    qubits = cudaq.qvector(qubit_count)
    h(qubits[0])
    for i in range(qubit_count - 1):
        x.ctrl(qubits[i], qubits[i + 1])
    mz(qubits)


counts = cudaq.sample(ghz, 3)
print(counts)
# [End Documentation]
