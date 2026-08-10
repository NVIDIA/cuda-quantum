# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[Begin Docs]
import cudaq
import math

cudaq.set_target("orca-photonics")


@cudaq.kernel
def kernel():
    n_modes = 2
    level = 3  # qudit level

    # Two qumode with 3 levels initialized to the ground / zero state.
    qumodes = [qudit(level) for _ in range(n_modes)]

    # Apply the create gate to the qumodes.
    for i in range(n_modes):
        create(qumodes[i])  # |00⟩ -> |11⟩

    # Apply the beam_splitter gate to the qumodes.
    beam_splitter(qumodes[0], qumodes[1], math.pi / 4)

    # Measurement operator.
    mz(qumodes)


# Sample the qumode for 1000 shots to gather statistics.
result = cudaq.sample(kernel)
print(result)
#[End Docs]
