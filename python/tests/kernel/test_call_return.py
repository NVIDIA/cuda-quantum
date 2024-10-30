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
    def callee(q: cudaq.qubit) -> bool:
        h(q)
        m = mz(q)
        return m

    @cudaq.kernel()
    def caller() -> bool:
        q = cudaq.qubit()
        t = callee(q)
        return t

    print(caller())


def test_call_with_return_bool2():
    from dataclasses import dataclass

    @dataclass
    class patch:
        data: cudaq.qview
        ancx: cudaq.qview
        ancz: cudaq.qview

    @cudaq.kernel()
    def stabilizer(logicalQubit: patch, x_stabilizers: list[int],
                   z_stabilizers: list[int]) -> bool:
        for xi in range(len(logicalQubit.ancx)):
            for di in range(len(logicalQubit.data)):
                if x_stabilizers[xi * len(logicalQubit.data) + di] == 1:
                    x.ctrl(logicalQubit.ancx[xi], logicalQubit.data[di])

        h(logicalQubit.ancx)
        for zi in range(len(logicalQubit.ancz)):
            for di in range(len(logicalQubit.data)):
                if z_stabilizers[zi * len(logicalQubit.data) + di] == 1:
                    x.ctrl(logicalQubit.data[di], logicalQubit.ancz[zi])

        results = mz(logicalQubit.ancx, logicalQubit.ancz)

        reset(logicalQubit.ancx)
        reset(logicalQubit.ancz)
        #TODO: support returning lists
        #Issue: https://github.com/NVIDIA/cuda-quantum/issues/2336
        return results[0] and results[2]

    @cudaq.kernel()
    def run() -> bool:
        q = cudaq.qvector(2)
        x(q[0])
        r = cudaq.qvector(2)
        s = cudaq.qvector(2)
        p = patch(q, r, s)

        return stabilizer(p, [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1])

    assert run() == True
