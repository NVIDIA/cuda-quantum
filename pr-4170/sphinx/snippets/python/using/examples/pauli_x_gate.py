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
    # A single qubit initialized to the ground / zero state.
    qubit = cudaq.qubit()

    # Apply the Pauli x gate to the qubit.
    x(qubit)

    # Measurement operator.
    mz(qubit)


# Sample the qubit for 1000 shots to gather statistics.
result = cudaq.sample(kernel)
print(result.most_probable())
#[End Docs]
