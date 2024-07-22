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
def kernel():
    qvector = cudaq.qvector(5)
    # Place qubits in GHZ State
    h(qvector[0])
    for qubit in range(4):
        x.ctrl(qvector[qubit], qvector[qubit + 1])


state_futures = []
for qpu in range(qpu_count):
    state_futures.append(cudaq.get_state_async(kernel, qpu_id=qpu))

for state in state_futures:
    print(state.get())
