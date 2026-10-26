# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. python3 %s

import cudaq

cudaq.set_target('quantinuum', machine='Helios-1SC', emulate=True)


@cudaq.kernel
def kernel(n_iter: int) -> bool:
    q0 = cudaq.qubit()
    for i in range(n_iter):
        h(q0)
        q0Result = mz(q0)
        if q0Result:
            break
    return mz(q0)


nShots = 100
nIter = 20
cudaq.set_random_seed(13)

results = cudaq.run(kernel, nIter, shots_count=nShots)
assert all(results)

cudaq.reset_target()
