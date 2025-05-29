# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

cudaq.set_target("quantinuum", emulate=True)
def test():
    @cudaq.kernel
    def kernel(n: int, m: np.int32):
        q = cudaq.qvector(n)
        j = 0
        jf = 1.2
        for i in range(10):
            k = 0 
            if i > 5:
                k = 1
            x(q[k])
            if mz(q[k]):
                j = j+ 1
                m = m + m
                jf = jf + jf
    
        if jf + 2 > n:
            x(q[0])
    
    
    print(cudaq.sample(kernel, 2, 134))

test()