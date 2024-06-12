# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. python3 %s --target quantinuum --emulate

import cudaq


@cudaq.kernel
def kernel(n_iter: int):
    q0 = cudaq.qubit()
    for i in range(n_iter):
        h(q0)
        q0Result = mz(q0)
        if q0Result:
            break


nShots = 100
nIter = 20
cudaq.set_random_seed(13)

counts = cudaq.sample(kernel, nIter, shots_count=nShots)
counts.dump()

assert len(counts.register_names) - 1 < nIter
