# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

cudaq.set_target("quantinuum", emulate=True)
def test():
    @cudaq.kernel
    def kernel(n: int):
        q = cudaq.qvector(n)
        for i in range(n):
            x(q[i])
        if mz(q[0]):
            return n + 1
        else:
            return n + 2

    
    print(cudaq.run(kernel, 2))

test()