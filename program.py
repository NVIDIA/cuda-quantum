# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

cudaq.reset_target()
cudaq.set_target('nvidia-fp64')

c = np.array([1. / np.sqrt(2.) + 0j, 1. / np.sqrt(2.), 0., 0.],
                dtype=cudaq.complex())
state = cudaq.State.from_data(c)

@cudaq.kernel
def kernel(vec: cudaq.State):
    q = cudaq.qvector(vec)

synthesized = cudaq.synthesize(kernel, state)
print(synthesized)
counts = cudaq.sample(synthesized)
print(counts)
assert '10' in counts
assert '00' in counts
