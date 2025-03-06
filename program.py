# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from typing import Callable

#cudaq.set_target('quantinuum', emulate=True)
# #cudaq.set_target("remote-mqpu", auto_launch="1")

# @cudaq.kernel
# def init():
#     q = cudaq.qvector(2)

# @cudaq.kernel
# def kernel(s: cudaq.State):
#     q = cudaq.qvector(s)


# state = cudaq.get_state(init)
# counts = cudaq.sample(kernel, state)
# print(counts)

cudaq.set_target("quantinuum", emulate = True)
#cudaq.set_target("remote-mqpu", auto_launch="1")

def test_grover():
    """Test that compute_action works in tandem with kernel composability."""

    @cudaq.kernel
    def reflect(qubits: cudaq.qview):
        ctrls = qubits.front(qubits.size() - 1)
        last = qubits.back()
        cudaq.compute_action(lambda: (h(qubits), x(qubits)),
                             lambda: z.ctrl(ctrls, last))

    @cudaq.kernel
    def oracle(q: cudaq.qview):
        z.ctrl(q[0], q[2])
        z.ctrl(q[1], q[2])

    print(reflect)

    @cudaq.kernel
    def grover(N: int, M: int, oracle: Callable[[cudaq.qview], None]):
        q = cudaq.qvector(N)
        h(q)
        for i in range(M):
            oracle(q)
            reflect(q)
        mz(q)

    print(grover)
    print(oracle)

    counts = cudaq.sample(grover, 3, 1, oracle)
    print(counts)
    #assert len(counts) == 2
    # assert '101' in counts
    # assert '011' in counts

test_grover()
