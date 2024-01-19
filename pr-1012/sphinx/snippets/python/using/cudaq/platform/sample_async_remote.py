# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
import cudaq

# Number of remote QPUs to launch
num_qpus = 2
cudaq.set_target("remote-mqpu",
                 remote_execution=True,
                 backend="tensornet",
                 auto_launch=str(num_qpus))
target = cudaq.get_target()
print("Number of QPUs:", target.num_qpus())

kernel, runtime_param = cudaq.make_kernel(int)
qubits = kernel.qalloc(runtime_param)
# Place qubits in superposition state.
kernel.h(qubits)
# Measure.
kernel.mz(qubits)

count_futures = []
for qpu in range(num_qpus):
    count_futures.append(cudaq.sample_async(kernel, 4 + qpu, qpu_id=qpu))

for counts in count_futures:
    print(counts.get())
