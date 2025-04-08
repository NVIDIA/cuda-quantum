# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

def test():

    @cudaq.kernel()
    def kernel() -> int:
        q = cudaq.qvector(2)
        x(q)
        return 6

    print(cudaq.run(kernel))
    print(cudaq.run(kernel, shots_count = 100))

#cudaq.set_target('quantinuum', emulate=True)
test()