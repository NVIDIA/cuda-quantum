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

kernel = cudaq.make_kernel()
qubits = kernel.qalloc(5)
# Place qubits in GHZ state.
kernel.h(qubits[0])
kernel.for_loop(0, 4, lambda i: kernel.cx(qubits[i], qubits[i + 1]))

state_futures = []
for qpu in range(num_qpus):
    state_futures.append(cudaq.get_state_async(kernel, qpu_id=qpu))

for state in state_futures:
    print(state.get())
