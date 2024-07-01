# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

c = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]

@cudaq.kernel
def kernel(vec: list[complex]):
    q = cudaq.qvector(vec)

synthesized = cudaq.synthesize(kernel, c)
print(synthesized)

counts = cudaq.sample(synthesized)
assert '00' in counts
assert '10' in counts