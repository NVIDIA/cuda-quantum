# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq


@cudaq.kernel
def kernel(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)
    h(qvector[0])
    for qubit in range(qubit_count - 1):
        x.ctrl(qvector[qubit], qvector[qubit + 1])
    mz(qvector)


#[Begin Sample1]
qubit_count = 2
print(cudaq.draw(kernel, qubit_count))
results = cudaq.sample(kernel, qubit_count)
# Should see a roughly 50/50 distribution between the |00> and
# |11> states. Example: {00: 505  11: 495}
print(results)
#[End Sample1]

#[Begin Sample2]
# With an increased shots count, we will still see the same 50/50 distribution,
# but now with 10,000 total measurements instead of the default 1000.
# Example: {00: 5005  11: 4995}
results = cudaq.sample(kernel, qubit_count, shots_count=10000)
print(results)
#[End Sample2]

#[Begin Sample3]
print(results.most_probable())  # prints: `00`
print(results.probability(results.most_probable()))  # prints: `0.5005`
#[End Sample3]
