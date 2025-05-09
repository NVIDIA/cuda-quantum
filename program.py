# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq import spin
import numpy as np

# @cudaq.kernel
# def ucc_single(qubits: cudaq.qview, words: list[list[cudaq.pauli_word]], theta: list[float]):
    
#     n_qubits = qubits.size()
    
    
#     for i in range(len(words)):
        
#         for term in words[i]:
#             exp_pauli(theta[i], qubits, term)
        
#         #exp_pauli(theta[i], qubits, words[i])

# @cudaq.kernel
# def kernel(qubit_num: int, words: list[list[cudaq.pauli_word]], theta: list[float]):
    
#     qubits = cudaq.qvector(qubit_num)
    
#     ucc_single(qubits, words, theta)
    
# #word = ['YZXI', 'XZYI']
# word = [[cudaq.pauli_word('YZXI'), cudaq.pauli_word('XZYI')], [cudaq.pauli_word('IYZX'), cudaq.pauli_word('IXZY')]]
# result = cudaq.sample(kernel, 4, word, [0.5, 0.5])
# print(result)


def test_list_creation():

    N = 10

    # @cudaq.kernel
    # def kernel(N: int, idx: int) -> int:
    #     myList = [i + 1 for i in range(N - 1)]
    #     return myList[idx]

    # for i in range(N - 1):
    #     assert kernel(N, i) == i + 1

    # @cudaq.kernel
    # def kernel2(N: int, i: int, j: int) -> int:
    #     myList = [[k, k] for k in range(N)]
    #     l = myList[i]
    #     return l[j]

    # print(kernel2(5, 0, 0))
    # for i in range(N):
    #     for j in range(2):
    #         print(i, j, kernel2(N, i, j))
    #         assert kernel2(N, i, j) == i

    # @cudaq.kernel
    # def kernel3(N: int):
    #     myList = list(range(N))
    #     q = cudaq.qvector(N)
    #     for i in myList:
    #         x(q[i])

    # print(kernel3)
    # counts = cudaq.sample(kernel3, 5)
    # assert len(counts) == 1
    # assert '1' * 5 in counts


    # @cudaq.kernel
    # def kernel4(myList: list[int]):
    #     q = cudaq.qvector(len(myList))
    #     cudaq.dbg.ast.print_i64(len(myList))
    #     casted = list(myList)
    #     for i in casted:
    #         cudaq.dbg.ast.print_i64(myList[i])
    #         x(q[i])

    # print(kernel4)
    # counts = cudaq.sample(kernel4, [0, 1, 2, 3])
    # print(counts)
    # assert len(counts) == 1
    # assert '1' * 4 in counts

    # @cudaq.kernel
    # def kernel4(myList: list[bool]):
    #     q = cudaq.qvector(len(myList))
    #     cudaq.dbg.ast.print_i64(len(myList))
    #     #casted = list(myList)
    #     for i in myList:
    #         cudaq.dbg.ast.print_i1(i)
    #         #x(q[i])

    # print(kernel4)
    # counts = cudaq.sample(kernel4, [True, False, True, False])
    # print(counts)
    # assert len(counts) == 1
    # #assert '1' * 4 in counts

    @cudaq.kernel
    def kernel5(myList: list[list[int]]):
        q = cudaq.qvector(len(myList))
        cudaq.dbg.ast.print_i64(len(myList))
        #casted = list(myList)
        #for i in myList:
        cudaq.dbg.ast.print_i64(myList[0][0])
            # for j in inner:
            #   cudaq.dbg.ast.print_i64(j)
            #x(q[i])

    print(kernel5)
    counts = cudaq.sample(kernel5, [[0,1,2,3], [4,5,6,7]])
    print(counts)
    assert len(counts) == 1
    assert '1' * 4 in counts

test_list_creation()