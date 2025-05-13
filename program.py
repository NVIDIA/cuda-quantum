# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

# def test_nested_list_iteration():

#     @cudaq.kernel
#     def kernel5(n: int, myList: list[list[list[bool]]]):
#         q = cudaq.qvector(n)
#         for k, inner in enumerate(myList):
#             for i, inner2 in enumerate(inner):
#                 for j, e in enumerate(inner2):
#                     if e:
#                         x(q[k*len(inner)*len(inner2) + i * len(inner2) + j])

#     counts = cudaq.sample(
#         kernel5, 16, [[[True, False, True, False], [False, True, False, True]], [[True, False, True, False], [False, True, False, True]]])
#     assert len(counts) == 1
#     print(counts)
#     assert '1010010110100101' in counts

# test_nested_list_iteration()

# def test_nested_list_iteration_int():
#     @cudaq.kernel
#     def kernel6(n: int, myList: list[list[list[int]]]):
#         q = cudaq.qvector(n)
#         for inner in myList:
#             for inner2 in inner:
#                 for i in inner2:
#                     x(q[i])

#     counts = cudaq.sample(kernel6, 16, [[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]])
#     assert len(counts) == 1
#     print(counts)
#     assert '1111111111111111' in counts

# test_nested_list_iteration_int()

# def test_nested_list_iteration_int4():
#     @cudaq.kernel
#     def kernel6(n: int, myList: list[list[list[list[int]]]]):
#         q = cudaq.qvector(n)
#         for inner in myList:
#             for inner2 in inner:
#                 for inner3 in inner2:
#                     for i in inner3:
#                         cudaq.dbg.ast.print_i64(i)
#                         x(q[i])

#     counts = cudaq.sample(kernel6, 16, [[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [10, 11]], [[12, 13], [14, 15]]]])
#     assert len(counts) == 1
#     print(counts)
#     assert '1111111111111111' in counts

# test_nested_list_iteration_int4()

def test_broadcast():
    @cudaq.kernel
    def kernel6(l: list[list[int]]):
        q = cudaq.qvector(2)
        for inner in l:
            for i in inner:
                cudaq.dbg.ast.print_i64(i)
                x(q[i])
        
    counts = cudaq.sample(kernel6, [[0, 1]])
    print(counts)

test_broadcast()