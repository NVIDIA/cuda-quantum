# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq

def test_state_synthesis():
    
    @cudaq.kernel
    def init(n: int):
        q = cudaq.qvector(n)
        h(q[0])
        x(q[1])
        mz(q)

    @cudaq.kernel
    def kernel(s: cudaq.State):
        q = cudaq.qvector(s)
        x(q[1])
        mz(q)

    s = cudaq.get_state(init, 2)
    s = cudaq.get_state(kernel, s)
    counts = cudaq.sample(kernel, s)
    print(counts)
    #assert "10" in counts
    #assert len(counts) == 1


cudaq.set_target("ionq", emulate = False)
test_state_synthesis()



