# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

cudaq.set_target('iqm', url="http://localhost/cocos", **{"qpu-architecture": "Adonis"})

@cudaq.kernel
def kernel(vec: list[complex]):
    qubits = cudaq.qvector(vec)

state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
counts = cudaq.sample(kernel, state)
print(counts)
assert '00' in counts
assert '10' in counts
assert not '01' in counts
assert not '11' in counts