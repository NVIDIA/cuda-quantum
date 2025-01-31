# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

def test_list_boundaries():

    # @cudaq.kernel
    # def kernel1():
    #     qubits = cudaq.qvector(2)
    #     r = range(0, 0)
    #     for i in r:
    #         x(qubits[i])

    # counts = cudaq.sample(kernel1)
    # assert len(counts) == 1
    # assert '00' in counts

    # @cudaq.kernel
    # def kernel2():
    #     qubits = cudaq.qvector(2)
    #     r = range(1, 0)
    #     for i in r:
    #         x(qubits[i])

    # counts = cudaq.sample(kernel2)
    # assert len(counts) == 1
    # assert '00' in counts

    @cudaq.kernel(verbose=True)
    def kernel3(n: int):
        qubits = cudaq.qvector(2)
        for i in range(n-1):
            x(qubits[i])

    counts = cudaq.sample(kernel3, 0)
    assert len(counts) == 1
    assert '00' in counts

    @cudaq.kernel
    def kernel4():
        qubits = cudaq.qvector(4)
        r = [i * 2 + 1 for i in range(-1)]
        for i in r:
            x(qubits[i])

    counts = cudaq.sample(kernel4)
    assert len(counts) == 1
    assert '0101' in counts

    # @cudaq.kernel
    # def kernel5():
    #     qubits = cudaq.qvector(4)
    #     r = [i * 2 + 1 for i in range(0)]
    #     for i in r:
    #         x(qubits[i])

    # counts = cudaq.sample(kernel5)
    # assert len(counts) == 1
    # assert '0000' in counts

test_list_boundaries()