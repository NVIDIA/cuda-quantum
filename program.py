# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq import spin
import numpy as np


def test_vector():
    cudaq.reset_target()
    cudaq.set_target("remote-mqpu", url="localhost:3030")
    #cudaq.set_target("quantinuum", emulate=True)

    c = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]

    @cudaq.kernel
    def kernel(r: int):
        q = cudaq.qvector(c)

    synthesized = cudaq.synthesize(kernel, 0)
    counts = cudaq.sample(synthesized)
    print(counts)

test_vector()

def test_state_from_data():
    cudaq.reset_target()
    cudaq.set_target("remote-mqpu", url="localhost:3030")
    #cudaq.set_target("quantinuum", emulate=True)


    c = np.array([1. / np.sqrt(2.),  1. / np.sqrt(2.), 0., 0.],
                    dtype=complex)
    print(cudaq.complex())
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel(s: cudaq.State):
        q = cudaq.qvector(s)

    counts = cudaq.sample(kernel, state)
    print(counts)

test_state_from_data()

def test_state_from_another_kernel():
    cudaq.reset_target()
    cudaq.set_target("remote-mqpu", url="localhost:3030")
    #cudaq.set_target("quantinuum", emulate=True)

    @cudaq.kernel
    def initState(n: int):
        q = cudaq.qvector(n)
        ry(np.pi/2, q[0])

    state = cudaq.get_state(initState, 2)

    @cudaq.kernel
    def kernel(s: cudaq.State):
        q = cudaq.qvector(s)

    counts = cudaq.sample(kernel, state)
    print(counts)

test_state_from_another_kernel()

#################################################
# @cudaq.kernel
# def kernel(angles: list[float], num_qubits: int):
#     qvector = cudaq.qvector(num_qubits)
#     x(qvector[0])
#     ry(angles[0], qvector[1])
#     x.ctrl(qvector[1], qvector[0])

# counts = cudaq.sample(kernel, [0.0, 1.0], 2)

# print(counts)