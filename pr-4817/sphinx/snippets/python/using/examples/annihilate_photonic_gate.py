# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[Begin Docs]
import cudaq

cudaq.set_target("orca-photonics")


@cudaq.kernel
def kernel():
    # A single qumode with 2 levels initialized to the ground / zero state.
    level = 2
    qumode = qudit(level)

    # Apply the create gate to the qumode.
    create(qumode)  # |0⟩ -> |1⟩

    # Apply the annihilate gate to the qumode.
    annihilate(qumode)  # |1⟩ -> |0⟩

    # Measurement operator.
    mz(qumode)


# Sample the qumode for 1000 shots to gather statistics.
# In this case, the results are deterministic and all return state 0.
result = cudaq.sample(kernel)
print(result)
#[End Docs]
