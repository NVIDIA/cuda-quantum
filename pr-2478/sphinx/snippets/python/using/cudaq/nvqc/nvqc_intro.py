# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
import cudaq

cudaq.set_target("nvqc")
num_qubits = 25
# Define a simple quantum kernel to execute on NVQC.
kernel = cudaq.make_kernel()
qubits = kernel.qalloc(num_qubits)
# Maximally entangled state between 25 qubits.
kernel.h(qubits[0])
for i in range(num_qubits - 1):
    kernel.cx(qubits[i], qubits[i + 1])
kernel.mz(qubits)

counts = cudaq.sample(kernel)
print(counts)
