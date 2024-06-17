# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import cudaq

import cudaq
import numpy as np

cudaq.reset_target()

cudaq.set_target('nvidia')
#cudaq.set_target('nvidia-mqpu')
# cudaq.set_target('density-matrix-cpu')


c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                dtype=np.complex128)
state = cudaq.State.from_data(c)

@cudaq.kernel(verbose=True)
def kernel(vec: cudaq.State):
    q = cudaq.qvector(vec)

print(kernel)
print(cudaq.to_qir(kernel))

#print(cudaq.get_target())
#counts = cudaq.sample(kernel, state)
#print(counts)