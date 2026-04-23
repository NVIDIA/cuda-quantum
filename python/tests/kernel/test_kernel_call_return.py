# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import cudaq


def test_call_with_callee_return_bool():

    @cudaq.kernel
    def bar(qubits: cudaq.qview) -> bool:
        x(qubits)
        return False

    @cudaq.kernel
    def foo(n: int):
        qubits = cudaq.qvector(n)
        bar(qubits)

    counts = cudaq.sample(foo, 3)
    assert "111" in counts and len(counts) == 1


def test_call_with_return_bool():

    @cudaq.kernel()
    def callee(q: cudaq.qubit) -> bool:
        x(q)
        m = mz(q)
        return m

    @cudaq.kernel()
    def caller() -> bool:
        q = cudaq.qubit()
        return callee(q)

    result = caller()
    assert result == True or result == False

    with pytest.raises(RuntimeError) as error:
        cudaq.sample(caller)
    assert ("The `sample` API only supports kernels that return None (void)"
            in repr(error))


def test_call_with_return_bool2():
    from dataclasses import dataclass

    @dataclass(slots=True)
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

        results = mz([*logicalQubit.ancx, *logicalQubit.ancz])

        reset(logicalQubit.ancx)
        reset(logicalQubit.ancz)
        #TODO: support returning lists
        #Issue: https://github.com/NVIDIA/cuda-quantum/issues/2336
        return results[3]

    @cudaq.kernel()
    def run() -> bool:
        q = cudaq.qvector(2)
        x(q[0])
        r = cudaq.qvector(2)
        s = cudaq.qvector(2)
        p = patch(q, r, s)

        return stabilizer(p, [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1])

    result = run()
    assert result == True or result == False

    with pytest.raises(RuntimeError) as error:
        cudaq.sample(run)
    assert "Kernel 'run' has return type '<class 'bool'>'" in repr(error)


def test_None_annotation():

    @cudaq.kernel
    def kernel() -> None:
        qubit = cudaq.qubit()
        h(qubit)

    # Test here is that this compiles
    cudaq.sample(kernel)
