# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
import cudaq

cudaq.set_target("nvidia-mqpu")
target = cudaq.get_target()
num_qpus = target.num_qpus()
print("Number of QPUs:", num_qpus)

@cudaq.kernel(jit=True)
def kernel(nr_qubits: int):

    qubits = cudaq.qvector(nr_qubits)
    # Place qubits in superposition state.
    kernel.h(qubits)
    # Measure.
    kernel.mz(qubits)


count_futures = []
for qpu in range(num_qpus):
    count_futures.append(cudaq.sample_async(kernel, 5, qpu_id=qpu))

for counts in count_futures:
    print(counts.get())
