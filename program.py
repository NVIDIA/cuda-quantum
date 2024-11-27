# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

@cudaq.kernel
def kernel_loop():
    numQubits = 5
    q = cudaq.qvector(numQubits)
    h(q)
    for i in range(4):
        cx(q[i], q[i + 1])
    for i in range(numQubits):
        mz(q[i])



asm = cudaq.translate(kernel_loop, format="openqasm2")
print(asm)

