# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

import cudaq

# Quantum Teleportation allows for the transfer of a quantum state from one 
# qubit to another, using a shared entangled pair and classical communication.

@cudaq.kernel
def teleportation():
    # Allocate 3 qubits:
    # q[0]: The qubit to be teleported (Alice's data)
    # q[1]: Alice's half of the Bell pair
    # q[2]: Bob's half of the Bell pair
    q = cudaq.qreg(3)

    # 1. Prepare the state to be teleported on q[0].
    # For this example, let's prepare the |1> state by applying X.
    x(q[0])

    # 2. Create a Bell pair between q[1] and q[2].
    h(q[1])
    x.ctrl(q[1], q[2])

    # 3. Alice performs a Bell measurement on q[0] and q[1].
    x.ctrl(q[0], q[1])
    h(q[0])

    # Mid-circuit measurement
    b0 = mz(q[0])
    b1 = mz(q[1])

    # 4. Bob applies conditional gates based on Alice's measurements.
    # In CUDA-Q, we can use if-statements on measurement results.
    if b1:
        x(q[2])
    if b0:
        z(q[2])

    # 5. Measure Bob's qubit to verify the state was teleported.
    # It should be in the |1> state.
    return mz(q[2])

# --- Execution ---

print("Executing Quantum Teleportation...")

# Since teleportation is a probabilistic process that requires 
# classical feedback, we use `cudaq.run` instead of `cudaq.sample`.
# `cudaq.sample` does not support kernels with conditional feedback.
shots_count = 100
results = cudaq.run(teleportation, shots_count=shots_count)

# Extract counts for the teleported qubit state
ones_count = sum(results)

print(f"Teleportation Results (Target Qubit):")
print(f"Measured '1': {ones_count} times out of {shots_count} shots.")

# Validation
if ones_count == shots_count:
    print("Success! The |1> state was teleported perfectly.")
else:
    print(f"Failure. Expected {shots_count} but got {ones_count}.")
