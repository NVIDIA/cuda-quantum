# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]
import cudaq

# Define our kernel.
kernel = cudaq.make_kernel()
# Allocate our qubits.
qubit_count = 2 
qvector = cudaq.qalloc(qubit_count)
# Place the first qubit in the superposition state.
kernel.h(qvector[0])
# Loop through the allocated qubits and apply controlled-X,
# or CNOT, operations between them.
for qubit in range(qubit_count-1):
    kernel.cx(qvector[qubit], qvector[qubit+1])
# Measure the qubits.
kernel.mz(qvector)
# [End Documentation]

# FIXME: Comment this back in when we're ready to roll out
# updated Python support.
# @cudaq.kernel
# def kernel(qubit_count: int):
#     # Allocate our qubits.
#     qvector = cudaq.qvector(qubit_count)
#     # Place the first qubit in the superposition state.
#     h(qvector[0])
#     # Loop through the allocated qubits and apply controlled-X,
#     # or CNOT, operations between them.
#     for qubit in range(qubit_count - 1):
#         x.ctrl(qvector[qubit], qvector[qubit + 1])
#     # Measure the qubits.
#     mz(qvector)


# Just for CI testing:
test_result = cudaq.sample(kernel, 10, shots_count=1)