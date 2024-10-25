# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq


def test_call_with_return_bool():

    @cudaq.kernel()
    def callMe(q: cudaq.qubit) -> bool:
        h(q)
        m = mz(q)
        return m

    @cudaq.kernel()
    def IWillCallYou() -> bool:
        q = cudaq.qubit()
        t = callMe(q)
        return t

    print(IWillCallYou())

def test_call_with_return_list():
    @cudaq.kernel(verbose=True)
    def callee(v: cudaq.qview) -> list[bool]:
        return mz(v)

    @cudaq.kernel()
    def caller() -> list[bool]:
        q = cudaq.qvector(2)
        t = callee(q)
        return t

    print(caller())

# def test_call_with_return_list():
#     from dataclasses import dataclass
#     @dataclass
#     class patch:
#         data : cudaq.qview 
#         ancx : cudaq.qview 
#         ancz : cudaq.qview 

# @cudaq.kernel(verbose=True)
# def stabilizer(logicalQubit: patch, x_stabilizers: list[int],
#                z_stabilizers: list[int]) -> list[bool]:
#     for xi in range(len(logicalQubit.ancx)):
#         for di in range(len(logicalQubit.data)):
#             if x_stabilizers[xi * len(logicalQubit.data) + di] == 1:
#                 x.ctrl(logicalQubit.ancx[xi], logicalQubit.data[di])

#     h(logicalQubit.ancx)
#     for zi in range(len(logicalQubit.ancx)):
#         for di in range(len(logicalQubit.data)):
#             if z_stabilizers[zi * len(logicalQubit.data) + di] == 1:
#                 x.ctrl(logicalQubit.data[di], logicalQubit.ancz[zi])

#     results = mz(logicalQubit.ancx, logicalQubit.ancz)

#     reset(logicalQubit.ancx)
#     reset(logicalQubit.ancz)
#     return results
