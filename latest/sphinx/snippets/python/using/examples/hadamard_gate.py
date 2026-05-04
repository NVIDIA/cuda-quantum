# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[Begin Docs]
import cudaq


@cudaq.kernel
def kernel():
    # A single qubit initialized to the ground/ zero state.
    qubit = cudaq.qubit()

    # Apply Hadamard gate to single qubit to put it in equal superposition.
    h(qubit)

    # Measurement operator.
    mz(qubit)


result = cudaq.sample(kernel)
print("Measured |0> with probability " +
      str(result["0"] / sum(result.values())))
print("Measured |1> with probability " +
      str(result["1"] / sum(result.values())))
#[End Docs]
