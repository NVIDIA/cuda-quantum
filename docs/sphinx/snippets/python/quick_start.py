# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]
import cudaq

@cudaq.kernel
def kernel():
    qubit = cudaq.qubit()
    h(qubit)
    mz(qubit)

result = cudaq.sample(kernel)
print(result)  # { 1:500 0:500 }