# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. python3 %s 
# RUN: PYTHONPATH=../../.. python3 %s --target quantinuum --emulate 

import cudaq

@cudaq.kernel(jit=True)
def kernel():
    q = cudaq.qvector(3)
    x(q[0])
    h(q[1])
    x.ctrl(q[1], q[2])
    x.ctrl(q[0], q[1])
    h(q[0])
    b0 = mz(q[0])
    b1 = mz(q[1])
    if b1:
        x(q[2])
    if b0:
        z(q[2])
    
    mz(q[2])

counts = cudaq.sample(kernel, shots_count=100)
counts.dump()
resultsOnZero = counts.get_marginal_counts([0])
resultsOnZero.dump()

nOnes = resultsOnZero.count('1')
assert nOnes == 100