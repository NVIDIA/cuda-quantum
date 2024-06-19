# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

@cudaq.kernel
def kernel():
    # 2 qubits both initialized to the ground/ zero state.
    qvector = cudaq.qvector(2)

    # Application of a flip gate to see ordering notation.
    x(qvector[0])

    mz(qvector[0])
    mz(qvector[1])

print(cudaq.draw(kernel))
result = cudaq.sample(kernel)
print(result)
