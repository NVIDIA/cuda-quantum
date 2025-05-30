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
def test_computations():
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
    
        if jf > 3 and j > 5:
            x(q[0])
    
    
    print(cudaq.sample(kernel, 2, 134))

test_computations()

def test_return():
    @cudaq.kernel
    def kernel(n: int) -> int:
        q = cudaq.qvector(n)
        if mz(q[0]):
            x(q[0])
            return 1
        return 0
    
    print(cudaq.sample(kernel, 2))

test_return()

# def test_branching():
#     @cudaq.kernel
#     def kernel(n: int) -> int:
#         q = cudaq.qvector(n)
#         match n:
#             case 0: return 1
#             case 1: return 2
#             case 0: return 3
#             case 1: return 4
#         return 5
    
#     print(cudaq.sample(kernel, 2))

# test_branching()