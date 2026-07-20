# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[Begin Observe1]
import cudaq
from cudaq import spin

operator = spin.z(0)
print(operator)  # prints: [1+0j] Z


@cudaq.kernel
def kernel():
    qubit = cudaq.qubit()
    h(qubit)
    # [End Observe1]


#[Begin Observe2]
result = cudaq.observe(kernel, operator)
print(result.expectation())  # prints: 0.0
#[End Observe2]

#[Begin Observe3]
result = cudaq.observe(kernel, operator, shots_count=1000)
print(result.expectation())  # prints non-zero value
#[End Observe3]
