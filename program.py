# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

def test_const_prop_loop_boundaries():
    @cudaq.kernel()
    def foo(q: cudaq.qview, b: int):
        for i in range(b):
            x(q[i])

    @cudaq.kernel()
    def kernel():
        qubits = cudaq.qvector(3)
        a = [0, 3]
        foo(qubits, a[1])

    cudaq.set_target("ionq", emulate=True)
    counts = cudaq.sample(kernel)
    print(counts)
    assert "111" in counts and len(counts) == 1

test_const_prop_loop_boundaries()


def test_const_prop_if_expressions():
    @cudaq.kernel()
    def foo(q: cudaq.qview, b: int):
        t = 2 if b > 0 else 3
        for i in range(t):
            x(q[i])


    @cudaq.kernel()
    def kernel():
        qubits = cudaq.qvector(3)
        a = [0, 3]
        foo(qubits, a[1])

    cudaq.set_target("ionq", emulate=True)
    counts = cudaq.sample(kernel)
    print(counts)
    assert "110" in counts and len(counts) == 1

#test_const_prop_if_expressions()

