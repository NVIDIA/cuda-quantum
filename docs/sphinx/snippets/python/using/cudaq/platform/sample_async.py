# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
import cudaq

cudaq.set_target("nvidia", option="mqpu")
target = cudaq.get_target()
qpu_count = target.num_qpus()
print("Number of QPUs:", qpu_count)


@cudaq.kernel
def kernel(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)
    # Place qubits in superposition state.
    h(qvector)
    # Measure.
    mz(qvector)


count_futures = []
for qpu in range(qpu_count):
    count_futures.append(cudaq.sample_async(kernel, 5, qpu_id=qpu))

for counts in count_futures:
    print(counts.get())
