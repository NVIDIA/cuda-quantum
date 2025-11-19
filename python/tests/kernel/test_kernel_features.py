# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest
import numpy as np
from typing import Callable, List
import sys

import cudaq
from cudaq import spin


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_argument_int():

    @cudaq.kernel
    def kernel(n: int):
        qubits = cudaq.qvector(n)

    counts = cudaq.sample(kernel, 2)
    assert len(counts) == 1
    assert '00' in counts

    @cudaq.kernel
    def kernel(n: np.int8):
        qubits = cudaq.qvector(n)

    counts = cudaq.sample(kernel, 2)
    assert len(counts) == 1
    assert '00' in counts

    @cudaq.kernel
    def kernel(n: np.int16):
        qubits = cudaq.qvector(n)

    counts = cudaq.sample(kernel, 2)
    assert len(counts) == 1
    assert '00' in counts

    @cudaq.kernel
    def kernel(n: np.int32):
        qubits = cudaq.qvector(n)

    counts = cudaq.sample(kernel, 2)
    assert len(counts) == 1
    assert '00' in counts

    @cudaq.kernel
    def kernel(n: np.int64):
        qubits = cudaq.qvector(n)

    counts = cudaq.sample(kernel, 2)
    assert len(counts) == 1
    assert '00' in counts

    @cudaq.kernel
    def kernel(n: np.int64):
        qubits = cudaq.qvector(n)

    counts = cudaq.sample(kernel, 2)
    assert len(counts) == 1
    assert '00' in counts


def test_adjoint():
    """Test that adjoint can be called on kernels and operations."""

    @cudaq.kernel
    def single_adjoint_test():
        q = cudaq.qubit()
        t(q)
        t.adj(q)

    counts = cudaq.sample(single_adjoint_test)
    assert '0' in counts
    assert len(counts) == 1

    @cudaq.kernel
    def qvector_adjoint_test():
        q = cudaq.qvector(2)
        t(q)
        t.adj(q)

    counts = cudaq.sample(qvector_adjoint_test)
    assert '00' in counts
    assert len(counts) == 1

    @cudaq.kernel
    def rotation_adjoint_test():
        q = cudaq.qubit()
        rx(1.1, q)
        rx.adj(1.1, q)

        ry(1.1, q)
        ry.adj(1.1, q)

    counts = cudaq.sample(rotation_adjoint_test)
    assert '0' in counts
    assert len(counts) == 1

    @cudaq.kernel
    def test_kernel_adjoint(q: cudaq.qview):
        h(q[0])
        t(q[1])
        s(q[2])

    @cudaq.kernel
    def test_caller():
        q = cudaq.qvector(3)
        x(q[0])
        x(q[2])
        test_kernel_adjoint(q)
        cudaq.adjoint(test_kernel_adjoint, q)

    counts = cudaq.sample(test_caller)
    assert len(counts) == 1
    assert '101' in counts


def test_control():
    """Test that we can control on kernel functions."""

    @cudaq.kernel
    def fancyCnot(a: cudaq.qubit, b: cudaq.qubit):
        x.ctrl(a, b)

    @cudaq.kernel
    def toffoli():
        q = cudaq.qvector(3)
        ctrl = q.front()
        # without a control, apply x to all
        x(ctrl, q[2])
        cudaq.control(fancyCnot, [ctrl], q[1], q[2])

    counts = cudaq.sample(toffoli)
    assert len(counts) == 1
    assert '101' in counts

    @cudaq.kernel
    def test():
        q, r, s = cudaq.qubit(), cudaq.qubit(), cudaq.qubit()
        x(q, s)
        swap.ctrl(q, r, s)

    counts = cudaq.sample(test)
    assert len(counts) == 1
    assert '110' in counts


def test_multi_control_gates():
    """Test that a multi-controlled X (Toffoli) gate works with the multi-control API."""

    @cudaq.kernel
    def mcx_kernel():
        q = cudaq.qvector(3)
        # State |110>
        x(q[0])
        x(q[1])
        x.ctrl([q[0], q[1]], q[2])
        mz(q)

    counts = cudaq.sample(mcx_kernel)
    assert len(counts) == 1
    assert '111' in counts

    @cudaq.kernel
    def mcx_kernel_neg():
        q = cudaq.qvector(3)
        # State |100>
        x(q[0])
        x.ctrl([q[0], q[1]], q[2])
        mz(q)

    counts = cudaq.sample(mcx_kernel_neg)
    assert len(counts) == 1
    assert '100' in counts


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
    assert len(counts) == 2
    assert '101' in counts
    assert '011' in counts


def test_2grover_compute_action():
    """Test that compute_action works in tandem with kernel composability."""

    @cudaq.kernel
    def reflect2(qubits: cudaq.qview):
        ctrls = qubits.front(qubits.size() - 1)
        last = qubits.back()

        def compute():
            h(qubits)
            x(qubits)

        # can also use
        # compute = lambda : (h(qubits), x(qubits))

        cudaq.compute_action(compute, lambda: z.ctrl(ctrls, last))

    print(reflect2)

    # Order matters, kernels must be "in-scope"
    @cudaq.kernel
    def oracle2(q: cudaq.qview):
        z.ctrl(q[0], q[2])
        z.ctrl(q[1], q[2])

    @cudaq.kernel
    def grover(N: int, M: int, oracle: Callable[[cudaq.qview], None]):
        q = cudaq.qvector(N)
        h(q)
        for i in range(M):
            oracle(q)
            reflect2(q)
        mz(q)

    # print(grover)

    counts = cudaq.sample(grover, 3, 1, oracle2)
    assert len(counts) == 2
    assert '101' in counts
    assert '011' in counts


def test_observe():

    @cudaq.kernel
    def ansatz():
        q = cudaq.qvector(1)

    molecule = 5.0 - 1.0 * spin.x(0)
    res = cudaq.observe(ansatz, molecule, shots_count=10000)
    assert np.isclose(res.expectation(), 5.0, atol=1e-1)


def test_pauli_word_input():

    h2_data = [
        3, 1, 1, 3, 0.0454063, 0, 2, 0, 0, 0, 0.17028, 0, 0, 0, 2, 0, -0.220041,
        -0, 1, 3, 3, 1, 0.0454063, 0, 0, 0, 0, 0, -0.106477, 0, 0, 2, 0, 0,
        0.17028, 0, 0, 0, 0, 2, -0.220041, -0, 3, 3, 1, 1, -0.0454063, -0, 2, 2,
        0, 0, 0.168336, 0, 2, 0, 2, 0, 0.1202, 0, 0, 2, 0, 2, 0.1202, 0, 2, 0,
        0, 2, 0.165607, 0, 0, 2, 2, 0, 0.165607, 0, 0, 0, 2, 2, 0.174073, 0, 1,
        1, 3, 3, -0.0454063, -0, 15
    ]
    h = cudaq.SpinOperator(h2_data, 4)

    @cudaq.kernel
    def kernel(theta: float, var: cudaq.pauli_word):
        q = cudaq.qvector(4)
        x(q[0])
        x(q[1])
        exp_pauli(theta, q, var)

    print(kernel)
    kernel(.11, 'XXXY')

    want_exp = cudaq.observe(kernel, h, .11, 'XXXY').expectation()
    assert np.isclose(want_exp, -1.13, atol=1e-2)

    want_exp = cudaq.observe(kernel, h, .11,
                             cudaq.pauli_word('XXXY')).expectation()
    assert np.isclose(want_exp, -1.13, atol=1e-2)

    @cudaq.kernel
    def test(theta: float, paulis: list[cudaq.pauli_word]):
        q = cudaq.qvector(4)
        x(q[0])
        x(q[1])
        for p in paulis:
            exp_pauli(theta, q, p)

    print(test)
    want_exp = cudaq.observe(test, h, .11, ['XXXY']).expectation()
    assert np.isclose(want_exp, -1.13, atol=1e-2)

    words = [cudaq.pauli_word('XXXY')]
    want_exp = cudaq.observe(test, h, .11, words).expectation()
    assert np.isclose(want_exp, -1.13, atol=1e-2)

    with pytest.raises(RuntimeError) as e:
        kernel(.11, 'HELLOBADTERM')


def test_exp_pauli():
    h2_data = [
        3, 1, 1, 3, 0.0454063, 0, 2, 0, 0, 0, 0.17028, 0, 0, 0, 2, 0, -0.220041,
        -0, 1, 3, 3, 1, 0.0454063, 0, 0, 0, 0, 0, -0.106477, 0, 0, 2, 0, 0,
        0.17028, 0, 0, 0, 0, 2, -0.220041, -0, 3, 3, 1, 1, -0.0454063, -0, 2, 2,
        0, 0, 0.168336, 0, 2, 0, 2, 0, 0.1202, 0, 0, 2, 0, 2, 0.1202, 0, 2, 0,
        0, 2, 0.165607, 0, 0, 2, 2, 0, 0.165607, 0, 0, 0, 2, 2, 0.174073, 0, 1,
        1, 3, 3, -0.0454063, -0, 15
    ]
    h = cudaq.SpinOperator(h2_data, 4)

    @cudaq.kernel
    def kernel(theta: float):
        q = cudaq.qvector(4)
        x(q[0])
        x(q[1])
        exp_pauli(theta, q, 'XXXY')

    print(kernel)
    want_exp = cudaq.observe(kernel, h, .11).expectation()
    assert np.isclose(want_exp, -1.13, atol=1e-2)


@pytest.mark.parametrize('target', ['default', 'stim'])
def test_dynamic_circuit(target):
    """Test that we correctly sample circuits with 
       mid-circuit measurements and conditionals."""

    if target == 'stim':
        save_target = cudaq.get_target()
        cudaq.set_target('stim')

    @cudaq.kernel
    def simple():
        q = cudaq.qvector(2)
        h(q[0])
        i = mz(q[0], register_name="c0")
        if i:
            x(q[1])
        mz(q)

    counts = cudaq.sample(simple, shots_count=100)
    counts.dump()
    c0 = counts.get_register_counts('c0')
    assert '0' in c0 and '1' in c0
    assert '00' in counts and '11' in counts

    @cudaq.kernel
    def simple():
        q = cudaq.qvector(2)
        h(q[0])
        i = mz(q[0])
        if i:
            x(q[1])
        mz(q)

    counts = cudaq.sample(simple)
    counts.dump()
    c0 = counts.get_register_counts('i')
    assert '0' in c0 and '1' in c0
    assert '00' in counts and '11' in counts

    if target == 'stim':
        cudaq.set_target(save_target)


def test_teleport():

    @cudaq.kernel
    def teleport():
        q = cudaq.qvector(3)
        x(q[0])
        h(q[1])

        x.ctrl(q[1], q[2])

        x.ctrl(q[0], q[1])
        h(q[0])

        b0 = mz(q[0])
        b1 = mz(q[1])

        if b1:
            x(q[2])

        if b0:
            z(q[2])

        mz(q[2])

    counts = cudaq.sample(teleport, shots_count=100)
    counts.dump()
    # Note this is testing that we can provide
    # the register name automatically
    b0 = counts.get_register_counts('b0')
    assert '0' in b0 and '1' in b0


def test_transitive_dependencies():

    @cudaq.kernel()
    def func0(q: cudaq.qubit):
        x(q)

    @cudaq.kernel()
    def func1(q: cudaq.qubit):
        func0(q)

    @cudaq.kernel
    def func2(q: cudaq.qubit):
        func1(q)

    @cudaq.kernel()
    def callMe():
        q = cudaq.qubit()
        func2(q)

    print(callMe)

    counts = cudaq.sample(callMe)
    assert len(counts) == 1 and '1' in counts

    # This test is for a bug where by
    # vqe_kernel thought kernel was a
    # dependency because cudaq.kernel
    # is a Call node in the AST.
    @cudaq.kernel
    def kernel():
        qubit = cudaq.qvector(2)
        h(qubit[0])
        x.ctrl(qubit[0], qubit[1])
        mz(qubit)

    result = cudaq.sample(kernel)
    print(result)
    assert len(result) == 2 and '00' in result and '11' in result

    @cudaq.kernel
    def vqe_kernel(nn: int):
        qubit = cudaq.qvector(nn)

        h(qubit[0])
        x.ctrl(qubit[0], qubit[1])

        mz(qubit)

    print(vqe_kernel)
    result = cudaq.sample(vqe_kernel, 2)
    print(result)
    assert len(result) == 2 and '00' in result and '11' in result


def test_decrementing_range():

    @cudaq.kernel
    def test(q: int, p: int):
        qubits = cudaq.qvector(5)
        for k in range(q, p, -1):
            cudaq.dbg.ast.print_i64(k)
            x(qubits[k])

    counts = cudaq.sample(test, 2, 0)
    counts.dump()
    assert '01100' in counts and len(counts) == 1

    @cudaq.kernel
    def test2(myList: List[int]):
        q = cudaq.qvector(len(myList))
        for i in range(0, len(myList), 2):
            cudaq.dbg.ast.print_i64(i)
            x(q[i])

    counts = cudaq.sample(test2, [0, 1, 2, 3])
    assert len(counts) == 1
    assert '1010' in counts


def test_no_dynamic_Lists():
    with pytest.raises(RuntimeError) as error:

        @cudaq.kernel
        def kernel(params: List[float]):
            params.append(1.0)

        kernel([])

    with pytest.raises(RuntimeError) as error:

        @cudaq.kernel
        def kernel():
            l = [i for i in range(10)]
            l.append(11)

        print(kernel)

    with pytest.raises(RuntimeError) as error:

        @cudaq.kernel
        def kernel():
            l = [[i, i, i] for i in range(10)]
            l.append([11, 12, 13])

        print(kernel)


def test_no_dynamic_lists():
    with pytest.raises(RuntimeError) as error:

        @cudaq.kernel
        def kernel(params: list[float]):
            params.append(1.0)

        print(kernel)


def test_simple_return_types():

    @cudaq.kernel
    def kernel(a: int, b: int) -> int:
        return a * b

    ret = kernel(2, 4)
    assert ret == 8

    @cudaq.kernel
    def kernel(a: int, b: int) -> np.int32:
        return a * b

    ret = kernel(2, 4)
    assert ret == 8

    @cudaq.kernel
    def kernel(a: int, b: int) -> np.int64:
        return a * b

    ret = kernel(2, 4)
    assert ret == 8

    @cudaq.kernel
    def kernel(a: float, b: float) -> float:
        return a * b

    ret = kernel(2, 4)
    assert np.isclose(ret, 8., atol=1e-12)

    @cudaq.kernel
    def kernel(a: float, b: float) -> np.float32:
        return a * b

    ret = kernel(2, 4)
    assert np.isclose(ret, 8., atol=1e-12)

    @cudaq.kernel
    def kernel(a: float, b: float) -> np.float64:
        return a * b

    ret = kernel(2, 4)
    assert np.isclose(ret, 8., atol=1e-12)

    with pytest.raises(RuntimeError) as error:

        @cudaq.kernel
        def kernel(a: int, b: int):  # No return type
            return a * b

    @cudaq.kernel
    def boolKernel() -> bool:
        return True

    assert boolKernel()


def test_tuple_creation_and_access():

    @cudaq.kernel
    def kernel() -> int:
        t = (-42, True, 13.5)
        return t[0]

    ret = kernel()
    assert ret == -42

    @cudaq.kernel
    def kernel() -> bool:
        t = (-42, True, 13.5)
        return t[1]

    ret = kernel()
    assert ret == True

    @cudaq.kernel
    def kernel() -> float:
        t = (-42, True, 13.5)
        return t[2]

    ret = kernel()
    assert np.isclose(ret, 13.5, atol=1e-12)

    @cudaq.kernel
    def kernel() -> float:
        t = (-42, True, 13.5)
        return t[3]

    with pytest.raises(RuntimeError) as e:
        ret = kernel()
    assert 'tuple index is out of range: 3' in repr(e)

    @cudaq.kernel
    def kernel(i: int) -> float:
        t = (-42, True, 13.5)
        return t[i]

    with pytest.raises(RuntimeError) as e:
        ret = kernel()
    assert 'non-constant subscript value on a tuple is not supported' in repr(e)


def test_list_creation():

    N = 10

    @cudaq.kernel
    def kernel(N: int, idx: int) -> int:
        myList = [i + 1 for i in range(N - 1)]
        return myList[idx]

    for i in range(N - 1):
        assert kernel(N, i) == i + 1

    @cudaq.kernel
    def kernel2(N: int, i: int, j: int) -> int:
        myList = [[k, k] for k in range(N)]
        l = myList[i]
        return l[j]

    print(kernel2(5, 0, 0))
    for i in range(N):
        for j in range(2):
            print(i, j, kernel2(N, i, j))
            assert kernel2(N, i, j) == i

    @cudaq.kernel
    def kernel3(N: int):
        myList = list(range(N))
        q = cudaq.qvector(N)
        for i in myList:
            x(q[i])

    counts = cudaq.sample(kernel3, 5)
    assert len(counts) == 1
    assert '1' * 5 in counts

    @cudaq.kernel
    def kernel4(myList: List[int]):
        q = cudaq.qvector(len(myList))
        casted = list(myList)
        for i in casted:
            x(q[i])

    counts = cudaq.sample(kernel4, list(range(5)))
    assert len(counts) == 1
    assert '1' * 5 in counts


def test_string_argument_error():

    @cudaq.kernel
    def kernel(n: int, s: str):
        qubits = cudaq.qvector(n)
        exp_pauli(2.2, qubits, 'YY')

    with pytest.raises(RuntimeError) as e:
        counts = cudaq.sample(kernel, "aaa")
    assert 'str is not a sup' in repr(e)


def test_list_string_argument_error():

    @cudaq.kernel
    def kernel(n: int, s: list[str]):
        qubits = cudaq.qvector(n)
        exp_pauli(2.2, qubits, 'YY')

    with pytest.raises(RuntimeError) as e:
        counts = cudaq.sample(kernel, 2, ["aaa"])
    assert 'str is not a sup' in repr(e)


def test_list_list_string_argument_error():

    @cudaq.kernel
    def kernel(n: int, s: list[list[str]]):
        qubits = cudaq.qvector(n)
        exp_pauli(2.2, qubits, 'YY')

    with pytest.raises(RuntimeError) as e:
        counts = cudaq.sample(kernel, 2, [["aaa"]])
    assert 'str is not a sup' in repr(e)


def test_broadcast():

    @cudaq.kernel
    def kernel(l: list[list[int]]):
        q = cudaq.qvector(2)
        for inner in l:
            for i in inner:
                x(q[i])

    #FIXME: update broadcast detection logic to allow this case.
    # https://github.com/NVIDIA/cuda-quantum/issues/2895
    with pytest.raises(RuntimeError) as e:
        counts = cudaq.sample(kernel, [[0, 1]])
    assert 'Invalid runtime argument type. Argument of type list[int] was provided' in repr(
        e)


def test_list_creation_with_cast():

    @cudaq.kernel
    def kernel(myList: list[int]):
        q = cudaq.qvector(len(myList))
        casted = list(myList)
        for i in casted:
            x(q[i])

    print(kernel)
    counts = cudaq.sample(kernel, list(range(5)))
    assert len(counts) == 1
    assert '1' * 5 in counts


def test_list_creation_with_cast():

    @cudaq.kernel
    def kernel(myList: list[int]):
        q = cudaq.qvector(len(myList))
        casted = list(myList)
        for i in casted:
            x(q[i])

    print(kernel)
    counts = cudaq.sample(kernel, list(range(5)))
    assert len(counts) == 1
    assert '1' * 5 in counts


def test_list_boundaries():

    @cudaq.kernel
    def kernel1():
        qubits = cudaq.qvector(2)
        r = range(0, 0)
        for i in r:
            x(qubits[i])

    counts = cudaq.sample(kernel1)
    assert len(counts) == 1
    assert '00' in counts

    @cudaq.kernel
    def kernel2():
        qubits = cudaq.qvector(2)
        r = range(1, 0)
        for i in r:
            x(qubits[i])

    counts = cudaq.sample(kernel2)
    assert len(counts) == 1
    assert '00' in counts

    @cudaq.kernel
    def kernel3():
        qubits = cudaq.qvector(2)
        for i in range(-1):
            x(qubits[i])

    counts = cudaq.sample(kernel3)
    assert len(counts) == 1
    assert '00' in counts

    @cudaq.kernel
    def kernel4():
        qubits = cudaq.qvector(4)
        r = [i * 2 + 1 for i in range(1)]
        for i in r:
            x(qubits[i])

    counts = cudaq.sample(kernel4)
    assert len(counts) == 1
    assert '0100' in counts

    @cudaq.kernel
    def kernel5():
        qubits = cudaq.qvector(4)
        r = [i * 2 + 1 for i in range(0)]
        for i in r:
            x(qubits[i])

    counts = cudaq.sample(kernel5)
    assert len(counts) == 1
    assert '0000' in counts

    @cudaq.kernel
    def kernel6():
        qubits = cudaq.qvector(4)
        r = [i * 2 + 1 for i in range(2)]
        for i in r:
            x(qubits[i])

    counts = cudaq.sample(kernel6)
    assert len(counts) == 1
    assert '0101' in counts

    @cudaq.kernel
    def kernel7():
        qubits = cudaq.qvector(5)
        r = [i for i in range(2, 5)]
        for i in r:
            x(qubits[i])

    counts = cudaq.sample(kernel7)
    assert len(counts) == 1
    assert '00111' in counts

    @cudaq.kernel
    def kernel8():
        qubits = cudaq.qvector(5)
        r = [i for i in range(2, 6, 2)]
        for i in r:
            x(qubits[i])

    counts = cudaq.sample(kernel8)
    assert len(counts) == 1
    assert '00101' in counts

    @cudaq.kernel
    def kernel9():
        qubits = cudaq.qvector(5)
        r = [i for i in range(6, 2, 2)]
        for i in r:
            x(qubits[i])

    counts = cudaq.sample(kernel9)
    assert len(counts) == 1
    assert '00000' in counts

    @cudaq.kernel
    def kernel10():
        qubits = cudaq.qvector(5)
        r = [i for i in range(3, 0, -2)]
        for i in r:
            x(qubits[i])

    counts = cudaq.sample(kernel10)
    assert len(counts) == 1
    assert '01010' in counts

    @cudaq.kernel
    def kernel11():
        qubits = cudaq.qvector(5)
        r = [i for i in range(-5, -2, -2)]
        for i in r:
            x(qubits[i])

    counts = cudaq.sample(kernel11)
    assert len(counts) == 1
    assert '00000' in counts

    @cudaq.kernel
    def kernel12():
        qubits = cudaq.qvector(5)
        r = [i for i in range(-1, -5, -2)]
        for i in r:
            x(qubits[-i])

    counts = cudaq.sample(kernel12)
    assert len(counts) == 1
    assert '01010' in counts

    @cudaq.kernel
    def kernel13():
        qubits = cudaq.qvector(5)
        r = [i for i in range(1, -4, -1)]
        for i in r:
            if i < 0:
                x(qubits[-i])
            else:
                x(qubits[i])

    counts = cudaq.sample(kernel13)
    assert len(counts) == 1
    assert '10110' in counts

    @cudaq.kernel
    def kernel14():
        qubits = cudaq.qvector(5)
        r = [i for i in range(-2, 6, 2)]
        for i in r:
            if i < 0:
                x(qubits[-i])
            else:
                x(qubits[i])

    counts = cudaq.sample(kernel14)
    assert len(counts) == 1
    assert '10001' in counts

    @cudaq.kernel
    def kernel15():
        qubits = cudaq.qvector(5)
        r = [i for i in range(1, 4, 0)]
        for i in r:
            x(qubits[i])

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel15)
    assert "range step value must be non-zero" in str(e.value)
    assert "offending source -> range(1, 4, 0)" in str(e.value)

    @cudaq.kernel
    def kernel16(v: int):
        qubits = cudaq.qvector(5)
        r = [i for i in range(1, 4, v)]
        for i in r:
            x(qubits[i])

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel16)
    assert "range step value must be a constant" in str(e.value)
    assert "offending source -> range(1, 4, v)" in str(e.value)


def test_array_value_assignment():

    @cudaq.kernel()
    def foo():
        a = [1, 1]
        b = [0, 0]
        b[0] = a[0]
        b[1] = a[1]
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        if (b[0]):
            x(q0)
        if (b[1]):
            x(q1)

    counts = cudaq.sample(foo)
    assert "11" in counts


def test_control_operations():

    @cudaq.kernel
    def test():
        q = cudaq.qvector(4)
        x.ctrl(q[0], q[1])
        cx(q[0], q[1])

    print(test)
    counts = cudaq.sample(test)


def test_control_operations():

    @cudaq.kernel
    def test(angle: float):
        q = cudaq.qvector(4)
        x.ctrl(q[0], q[1])
        cx(q[0], q[1])
        rx.ctrl(angle, q[0], q[1])
        crx(angle, q[0], q[1])

    print(test)
    counts = cudaq.sample(test, 0.785398)


def test_bool_op_short_circuit():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        if mz(qubits[0]) and mz(qubits[1]):
            x(qubits[1])
        mz(qubits[1])

    print(kernel)

    counts = cudaq.sample(kernel)
    counts.dump()
    assert len(counts) == 2 and '10' in counts and '00' in counts


def test_sample_async_issue_args_processed():

    @cudaq.kernel
    def kernel(params: np.ndarray):
        q = cudaq.qvector(2)
        x(q[0])
        ry(params[0], q[1])
        x.ctrl(q[1], q[0])

    params = np.array([.59])
    result = cudaq.sample_async(kernel, params, qpu_id=0)
    counts = result.get()
    assert len(counts) == 2 and '01' in counts and '10' in counts


def test_capture_vars():

    n = 5
    f = 0.0
    m = 5
    hello = str()

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(n)
        x(q)
        cudaq.dbg.ast.print_f64(f)
        for qb in q:
            rx(f, qb)

    counts = cudaq.sample(kernel)
    counts.dump()
    assert '1' * n in counts

    f = np.pi
    counts = cudaq.sample(kernel)
    counts.dump()
    assert '0' * n in counts

    n = 7
    f = 0.0
    counts = cudaq.sample(kernel)
    counts.dump()
    assert '1' * n in counts

    counts = cudaq.sample(kernel)
    counts.dump()
    assert '1' * n in counts

    n = 3
    counts = cudaq.sample(kernel)
    counts.dump()
    assert '1' * n in counts

    @cudaq.kernel
    def testCanOnlyCaptureIntAndFloat():
        i = hello

    with pytest.raises(RuntimeError) as e:
        testCanOnlyCaptureIntAndFloat()

    b = True

    @cudaq.kernel
    def canCaptureBool():
        q = cudaq.qubit()
        if b:
            x(q)

    counts = cudaq.sample(canCaptureBool)
    counts.dump()
    assert len(counts) == 1 and '1' in counts

    b = False
    counts = cudaq.sample(canCaptureBool)
    counts.dump()
    assert len(counts) == 1 and '0' in counts

    l = [.59]
    li = [0, 1]

    @cudaq.kernel
    def canCaptureList():
        q = cudaq.qvector(2)
        firstIdx = li[0]
        secondIdx = li[1]
        x(q[firstIdx])
        ry(l[firstIdx], q[secondIdx])
        x.ctrl(q[secondIdx], q[firstIdx])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    assert np.isclose(-1.748,
                      cudaq.observe(canCaptureList, hamiltonian).expectation(),
                      atol=1e-3)


def test_inner_function_capture():

    n = 3
    m = 5

    def innerClassical():

        @cudaq.kernel()
        def foo():
            q = cudaq.qvector(n)

        def innerInnerClassical():

            @cudaq.kernel()
            def bar():
                q = cudaq.qvector(m)
                x(q)

            return cudaq.sample(bar)

        return cudaq.sample(foo), innerInnerClassical()

    fooCounts, barCounts = innerClassical()
    assert len(fooCounts) == 1 and '0' * n in fooCounts
    assert len(barCounts) == 1 and '1' * m in barCounts


def test_error_qubit_constructor():

    @cudaq.kernel
    def test():
        q = cudaq.qubit(10)
        h(q[0])

    with pytest.raises(RuntimeError) as e:
        test.compile()


def test_swallow_measure_value():

    @cudaq.kernel
    def test():
        data = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        mz(ancilla)
        x(data[1])

    # The test here is that this compiles.
    test.compile()
    print(test)


def test_compare_with_true():

    @cudaq.kernel
    def test():
        data = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        results = mz(ancilla)
        if results[0] == True:
            x(data[0])

    # The test here is that this compiles.
    test()


def test_with_docstring():

    @cudaq.kernel
    def oracle(register: cudaq.qvector, auxillary_qubit: cudaq.qubit,
               hidden_bitstring: List[int]):
        """
        The inner-product oracle for the Bernstein-Vazirani algorithm.
        """
        for index, bit in enumerate(hidden_bitstring):
            if bit == 0:
                # Apply identity operation to the qubit if it's
                # in the 0-state.
                # In this case, we do nothing.
                pass
            else:
                # Otherwise, apply a `cx` gate with the current qubit as
                # the control and the auxillary qubit as the target.
                cx(register[index], auxillary_qubit)

    @cudaq.kernel
    def bernstein_vazirani(qubit_count: int, hidden_bitstring: List[int]):
        """
        Returns a kernel implementing the Bernstein-Vazirani algorithm
        for a random, hidden bitstring.
        """
        # Allocate the specified number of qubits - this
        # corresponds to the length of the hidden bitstring.
        qubits = cudaq.qvector(qubit_count)
        # Allocate an extra auxillary qubit.
        auxillary_qubit = cudaq.qubit()

        # Prepare the auxillary qubit.
        h(auxillary_qubit)
        z(auxillary_qubit)

        # Place the rest of the register in a superposition state.
        h(qubits)

        # Query the oracle.
        oracle(qubits, auxillary_qubit, hidden_bitstring)

        # Apply another set of Hadamards to the register.
        h(qubits)

        # Apply measurement gates to just the `qubits`
        # (excludes the auxillary qubit).
        mz(qubits)

    # Test here is that it compiles
    bernstein_vazirani.compile()


def test_disallow_list_no_element_type():

    @cudaq.kernel
    def test(listVar: List):
        pass

    with pytest.raises(RuntimeError) as e:
        print(test)


def test_invalid_cudaq_type():

    @cudaq.kernel
    def test():
        q = cudaq.qreg(5)
        h(q)

    with pytest.raises(RuntimeError) as e:
        print(test)


def test_bool_list_elements():

    @cudaq.kernel
    def kernel(var: list[bool]):
        q = cudaq.qubit()
        x(q)
        if var[0]:
            x(q)

    counts = cudaq.sample(kernel, [False], shots_count=100)
    assert '1' in counts and len(counts) == 1

    counts = cudaq.sample(kernel, [True], shots_count=100)
    assert '0' in counts and len(counts) == 1


def test_list_float_pass_list_int():

    @cudaq.kernel
    def test(var: List[float]):
        q = cudaq.qvector(2)
        cudaq.dbg.ast.print_f64(var[0])
        x(q[int(var[0])])
        x(q[int(var[1])])

    var = [0, 1]
    counts = cudaq.sample(test, var)
    assert len(counts) == 1 and '11' in counts
    counts.dump()


def test_cmpi_error_ints_different_widths():

    @cudaq.kernel
    def test():
        q = cudaq.qubit()
        i = mz(q)
        if i == 1:
            x(q)

    test()
    counts = cudaq.sample(test)
    assert '0' in counts and len(counts) == 1


def test_aug_assign_add():

    @cudaq.kernel
    def test() -> float:
        f = 5.
        f += 5.
        return f

    assert test() == 10.

    @cudaq.kernel
    def test2() -> int:
        i = 5
        i += 5
        return i

    assert test2() == 10


def test_empty_lists():

    @cudaq.kernel
    def empty(var: list[cudaq.pauli_word], varvar: list[float],
              varvarvar: list[bool]):
        q = cudaq.qvector(2)
        x(q[0])

    empty([], [], [])


def test_no_valueerror_np_array():

    @cudaq.kernel
    def test(var: np.ndarray):
        q = cudaq.qubit()
        ry(var[0], q)
        mz(q)

    test(np.array([1., 2.]))


def test_draw():

    @cudaq.kernel
    def kernel_to_draw():
        q = cudaq.qvector(4)
        h(q)
        # Broadcast
        cx(q[0], q[1])
        cy([q[0], q[1]], q[2])
        cy([q[2], q[0]], q[1])
        cy([q[1], q[2]], q[0])
        z(q[2])

        swap(q[0], q[2])
        swap(q[1], q[2])
        swap(q[0], q[1])
        swap(q[0], q[2])
        swap(q[1], q[2])

        r1(3.14159, q[0])
        tdg(q[1])
        s(q[2])

    circuit = cudaq.draw(kernel_to_draw)
    print(circuit)
    expected_str = '''     ╭───╮               ╭───╮                 ╭───────────╮       
q0 : ┤ h ├──●────●────●──┤ y ├──────╳─────╳──╳─┤ r1(3.142) ├───────
     ├───┤╭─┴─╮  │  ╭─┴─╮╰─┬─╯      │     │  │ ╰───────────╯╭─────╮
q1 : ┤ h ├┤ x ├──●──┤ y ├──●────────┼──╳──╳──┼───────╳──────┤ tdg ├
     ├───┤╰───╯╭─┴─╮╰─┬─╯  │  ╭───╮ │  │     │       │      ╰┬───┬╯
q2 : ┤ h ├─────┤ y ├──●────●──┤ z ├─╳──╳─────╳───────╳───────┤ s ├─
     ├───┤     ╰───╯          ╰───╯                          ╰───╯ 
q3 : ┤ h ├─────────────────────────────────────────────────────────
     ╰───╯                                                         
'''

    assert circuit == expected_str


def test_draw_fail():

    @cudaq.kernel
    def kernel(argument: float):
        q = cudaq.qvector(2)
        h(q[0])
        ry(argument, q[1])

    with pytest.raises(RuntimeError) as error:
        print(cudaq.draw(kernel))


def test_draw_bug_1400():

    @cudaq.kernel
    def bell_pair():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])
        mz(q)

    @cudaq.kernel
    def kernel(angle: float):
        q = cudaq.qubit()
        h(q)
        ry(angle, q)

    print(cudaq.draw(kernel, 0.59))
    print(cudaq.draw(kernel, 0.59))
    circuit = cudaq.draw(bell_pair)
    print(circuit)
    expected_str = '''     ╭───╮     
q0 : ┤ h ├──●──
     ╰───╯╭─┴─╮
q1 : ─────┤ x ├
          ╰───╯
'''
    assert circuit == expected_str


def test_with_docstring_2():

    @cudaq.kernel
    def simple(n: int):
        '''
        A docstring with triple single quote
        '''
        qubits = cudaq.qvector(n)
        exp_pauli(2.2, qubits, 'YYYY')
        """
        A docstring in the middle of kernel
        """
        for q in qubits:
            '''
            A multi-line string.
            Should be ignored.
            '''
            h(q)

    @cudaq.kernel
    def kernel():
        simple(4)

    kernel.compile()
    print(kernel)


def test_user_error_op_attr_1446():

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        x(qubits)
        x.control(qubits[0], qubits[1])
        h(qubits)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'Unknown attribute on quantum' in repr(
        e) and 'Did you mean x.ctrl(...)?' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        x(qubits)
        x.adjoint(qubits[0], qubits[1])
        h(qubits)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'Unknown attribute on quantum' in repr(
        e) and 'Did you mean x.adj(...)?' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        x(qubits)
        x.adjointBadAttr(qubits[0], qubits[1])
        h(qubits)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'Unknown attribute on quantum' in repr(
        e) and 'Did you mean x.adj(...)?' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        x(qubits)
        x.noIdeaWhatThisIs(qubits[0], qubits[1])
        h(qubits)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'Unknown attribute on quantum' in repr(
        e) and 'Did you mean ' not in repr(e)


def test_ctrl_wrong_dtype_1447():

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        x(qubits)
        # should throw error for acting on ints
        x.ctrl(0, 1)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'invalid argument type for control operand' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        x(qubits)
        # should throw error for acting on ints
        x.ctrl(qubits[0], 1)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'invalid argument type for target operand' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        x(qubits)
        # should throw error for acting on ints
        swap(0, 1)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'invalid argument type for target operand' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        cx(0, 1)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'invalid argument type for control operand' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        h(22)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'invalid argument type for target operand' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        crx(2.2, 0, 1)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'invalid argument type for control operand' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        mz(22)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'invalid argument type for target operand' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        rx.ctrl(2.2, 2, 3)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'invalid argument type for control operand' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        sdg(2)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'invalid argument type for target operand' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        x.adj(3)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'invalid argument type for target operand' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        swap.ctrl(2, 3, 4)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'invalid argument type for control operand' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        rx.ctrl(1.1, 3, 2)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'invalid argument type for control operand' in repr(e)

    @cudaq.kernel
    def test_kernel(nQubits: int):
        qubits = cudaq.qvector(nQubits)
        rx.adj(2.2, 3)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'invalid argument type for target operand' in repr(e)


def test_math_module_pi_1448():
    import math

    @cudaq.kernel
    def test_kernel() -> float:
        theta = math.pi
        return theta

    test_kernel.compile()
    assert np.isclose(test_kernel(), math.pi, 1e-12)


def test_len_qvector_1449():

    @cudaq.kernel
    def test_kernel(nCountingQubits: int) -> int:
        qubits = cudaq.qvector(nCountingQubits)
        # can use N = counting_qubits.size()
        N = len(qubits)
        h(qubits)
        return N

    test_kernel.compile()
    assert test_kernel(5) == 5


def test_missing_paren_1450():

    @cudaq.kernel
    def test_kernel():
        state_reg = cudaq.qubit
        x(state_reg)

    with pytest.raises(RuntimeError) as e:
        test_kernel.compile()
    assert 'no valid value was created' in repr(e)
    assert '(offending source -> state_reg = cudaq.qubit)' in repr(e)


def test_cast_error_1451():

    @cudaq.kernel
    def test_kernel(N: int) -> float:
        return N / 2

    # Test is that this compiles
    out = test_kernel(5)
    assert out == 2.5

    @cudaq.kernel
    def test_kernel(N: int):
        q = cudaq.qvector(N)
        # range fails for non-integer values,
        # matching Python behavior
        for i in range(0, int(N / 2)):
            swap(q[i], q[N - i - 1])

    # Test is that this compiles
    test_kernel.compile()


def test_bad_attr_call_error():

    @cudaq.kernel
    def test_state(N: int):
        q = cudaq.qvector(N)
        h(q[0])
        kernel.h(q[0])

    with pytest.raises(RuntimeError) as e:
        test_state.compile()
    assert "unknown function call" in repr(e)
    assert "offending source -> kernel.h(q[0])" in repr(e)


def test_bad_return_value_with_stdvec_arg():

    @cudaq.kernel
    def test_param(i: int, l: List[int]) -> int:
        return i

    l = [42]
    for i in range(4):
        assert test_param(i, l) == i


def test_bad_return_int_bool_param():

    @cudaq.kernel
    def kernel(c: int, b: bool) -> int:
        return c

    assert kernel(1, False) == 1


def test_return_bool_bool_param():

    @cudaq.kernel
    def kernel(b: bool, b2: bool) -> bool:
        return b

    assert kernel(True, False) == True


def test_return_int_int_param():

    @cudaq.kernel
    def kernel(b: int, b2: int) -> int:
        return b

    assert kernel(42, 53) == 42


def test_return_no_param():

    @cudaq.kernel
    def kernel() -> int:
        return 42

    assert kernel() == 42


def test_no_param_no_return():

    @cudaq.kernel
    def kernel():
        return

    kernel()


def test_measure_variadic_qubits():

    @cudaq.kernel
    def test():
        q = cudaq.qvector(5)
        x(q[2])
        mz(q[0], q[1], q[2])

    counts = cudaq.sample(test)
    assert len(counts) == 1 and '001' in counts

    @cudaq.kernel
    def test():
        q = cudaq.qvector(5)
        x(q[0], q[2])
        mz([q[0], *q[1:3]])

    counts = cudaq.sample(test)
    assert len(counts) == 1 and '101' in counts


def test_bad_return_value_with_stdvec_arg():

    @cudaq.kernel
    def test_param(i: int, l: List[int]) -> int:
        return i

    l = [42]
    for i in range(4):
        assert test_param(i, l) == i


def test_u3_op():

    @cudaq.kernel
    def check_x():
        q = cudaq.qubit()
        # implement Pauli-X gate with U3
        u3(np.pi, np.pi, np.pi / 2, q)

    print(check_x)
    counts = cudaq.sample(check_x)
    assert counts["1"] == 1000

    @cudaq.kernel
    def bell_pair():
        qubits = cudaq.qvector(2)
        # implement Hadamard gate with U3
        u3(np.pi / 2, 0, np.pi, qubits[0])
        cx(qubits[0], qubits[1])

    counts = cudaq.sample(bell_pair)
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)


def test_u3_ctrl():

    @cudaq.kernel
    def another_bell_pair():
        qubits = cudaq.qvector(2)
        u3(np.pi / 2, 0, np.pi, qubits[0])
        u3.ctrl(np.pi, np.pi, np.pi / 2, qubits[0], qubits[1])

    print(another_bell_pair)
    counts = cudaq.sample(another_bell_pair)
    assert (len(counts) == 2)
    assert ('00' in counts)
    assert ('11' in counts)


def test_u3_adj():

    @cudaq.kernel
    def rotation_adjoint_test():
        q = cudaq.qubit()

        # implement Rx gate with U3
        u3(1.1, -np.pi / 2, np.pi / 2, q)
        # rx.adj(angle) = u3.adj(angle, -pi/2, pi/2)
        u3.adj(1.1, -np.pi / 2, np.pi / 2, q)

        # implement Ry gate with U3
        u3(1.1, 0, 0, q)
        u3.adj(1.1, 0, 0, q)

    print(rotation_adjoint_test)

    counts = cudaq.sample(rotation_adjoint_test)
    assert '0' in counts
    assert len(counts) == 1


def test_u3_parameterized():

    @cudaq.kernel
    def param_kernel(theta: float, phi: float, lambda_: float):
        q = cudaq.qubit()
        u3(theta, phi, lambda_, q)

    counts = cudaq.sample(param_kernel, np.pi, np.pi, np.pi / 2)
    assert counts["1"] == 1000


def test_reset():

    @cudaq.kernel
    def single_qubit():
        q = cudaq.qubit()
        x(q)
        reset(q)

    counts = cudaq.sample(single_qubit)
    assert counts['0'] == 1000

    @cudaq.kernel
    def multiple_qubits(num_iters: int) -> int:
        q = cudaq.qvector(2)
        nCorrect = 0
        for i in range(num_iters):
            h(q[0])
            x.ctrl(q[0], q[1])
            results = mz(q)
            if results[0] == results[1]:
                nCorrect = nCorrect + 1

            reset(q)
        return nCorrect

    counts = multiple_qubits(100)
    print(f'N Correct = {counts}')
    assert counts == 100


def test_nested_loops_with_break():

    @cudaq.kernel
    def prog(theta: float):
        q = cudaq.qvector(2)

        for _ in range(5):
            while True:
                x(q)
                ry(theta, q[1])
                res = mz(q[1])

                if res:
                    x(q[1])
                    break
        mz(q)

    # The test here is that this compiles.
    prog.compile()
    print(prog)


def test_nested_loops_with_continue():

    @cudaq.kernel
    def prog():
        q = cudaq.qvector(10)
        j = 0
        for num in range(2, 10):
            while j < num:
                if num % 2 == 0:
                    h(q[num])
                    continue
                x(q[num])
            j += 1

    # The test here is that this compiles.
    prog.compile()
    print(prog)


def test_issue_1682():

    @cudaq.kernel
    def qrbm_reuse_ancilla(v_nodes: int, h_nodes: int, theta: list[float],
                           coupling: list[float]):

        qubits_num = v_nodes + h_nodes
        qubits = cudaq.qvector(qubits_num)
        ancilla = cudaq.qubit()

        count = 0
        for i in range(v_nodes + h_nodes):
            ry(theta[count], qubits[i])
            count += 1

        count = 0

        for v in range(v_nodes):
            for h in range(v_nodes, v_nodes + h_nodes):

                while True:
                    ry.ctrl(coupling[count], qubits[v], qubits[h], ancilla)
                    x(qubits[v])
                    ry.ctrl(coupling[count + 1], qubits[v], qubits[h], ancilla)
                    x(qubits[v])
                    x(qubits[h])
                    ry.ctrl(coupling[count + 1], qubits[v], qubits[h], ancilla)
                    x(qubits[v])
                    ry.ctrl(coupling[count], qubits[v], qubits[h], ancilla)
                    x(qubits[v])
                    x(qubits[h])

                    res = mz(ancilla)

                    if res:
                        x(ancilla)
                        break

                count += 2

        mz(qubits)

    qrbm_reuse_ancilla.compile()


def test_subtract():

    @cudaq.kernel
    def bug_subtract():
        qubits = cudaq.qvector(4)
        x(qubits[0:2])
        mu = 0.7951
        sigma = 0.6065
        rz(1.0 - (mu / sigma), qubits[1])
        mz(qubits)

    cudaq.sample(bug_subtract)


def test_capture_opaque_kernel():

    def retFunc():

        @cudaq.kernel
        def bell(i: int):
            q = cudaq.qvector(i)
            h(q[0])
            x.ctrl(q[0], q[1])

        return bell

    def retFunc2():

        @cudaq.kernel
        def super():
            q = cudaq.qubit()
            h(q)

        return super

    b = retFunc()

    @cudaq.kernel
    def k():
        b(2)

    print(k)

    b = retFunc2()

    @cudaq.kernel
    def kd():
        b()

    print(kd)

    counts = cudaq.sample(k)
    assert len(counts) == 2 and '00' in counts and '11' in counts

    counts = cudaq.sample(kd)
    assert len(counts) == 2 and '0' in counts and '1' in counts


def test_custom_classical_kernel_type():
    from dataclasses import dataclass

    @dataclass(slots=True)
    class CustomIntAndFloatType:
        integer: int
        floatingPoint: float

    instance = CustomIntAndFloatType(123, 123.123)
    assert instance.integer == 123 and instance.floatingPoint == 123.123

    @cudaq.kernel
    def test(input: CustomIntAndFloatType):
        qubits = cudaq.qvector(input.integer)
        ry(input.floatingPoint, qubits[0])
        rx(input.floatingPoint * 2, qubits[0])
        x.ctrl(qubits[0], qubits[1])

    instance = CustomIntAndFloatType(2, np.pi / 2.)
    counts = cudaq.sample(test, instance)
    counts.dump()
    assert len(counts) == 2 and '00' in counts and '11' in counts

    # FIXME:
    # While this exact test worked, the handing in OpaqueArguments.h
    # does not match the expected layout in the args creator.
    # Correspondingly, both subsequent tests below failed with a crash
    # as it was. I hence choose to give a proper error until this is
    # fixed after general Python compiler revisions.
    @dataclass(slots=True)
    class CustomIntAndListFloat:
        integer: int
        array: list[float]

    @cudaq.kernel
    def test(input: CustomIntAndListFloat):
        qubits = cudaq.qvector(input.integer)
        ry(input.array[0], qubits[0])
        ry(input.array[1], qubits[2])
        x.ctrl(qubits[0], qubits[1])

    instance = CustomIntAndListFloat(3, [np.pi, np.pi])
    with pytest.raises(RuntimeError) as e:
        counts = cudaq.sample(test, instance)
    assert 'dynamically sized element types for function arguments are not yet supported' in str(
        e.value)
    # Should be: assert len(counts) == 1 and '111' in counts

    @cudaq.kernel
    def test(input: CustomIntAndListFloat):
        qubits = cudaq.qvector(input.integer)
        ry(input.array[1], qubits[0])
        ry(input.array[3], qubits[2])
        x.ctrl(qubits[0], qubits[1])

    instance = CustomIntAndListFloat(3, [0, np.pi, 0, np.pi])
    with pytest.raises(RuntimeError) as e:
        counts = cudaq.sample(test, instance)
    assert 'dynamically sized element types for function arguments are not yet supported' in str(
        e.value)
    # Should be: assert len(counts) == 1 and '111' in counts

    @dataclass(slots=True)
    class CustomIntAndListFloat:
        array: list[float]
        integer: int

    @cudaq.kernel
    def test(input: CustomIntAndListFloat):
        qubits = cudaq.qvector(input.integer)
        ry(input.array[1], qubits[0])
        ry(input.array[3], qubits[2])
        x.ctrl(qubits[0], qubits[1])

    instance = CustomIntAndListFloat([0, np.pi, 0, np.pi], 3)
    with pytest.raises(RuntimeError) as e:
        counts = cudaq.sample(test, instance)
    assert 'dynamically sized element types for function arguments are not yet supported' in str(
        e.value)
    # Should be: assert len(counts) == 1 and '111' in counts

    @dataclass(slots=True)
    class CustomIntAndListFloat:
        integer: list[int]
        array: list[int]

    @cudaq.kernel
    def test(input: CustomIntAndListFloat):
        qubits = cudaq.qvector(input.integer[-1])
        ry(input.array[1], qubits[0])
        ry(input.array[3], qubits[2])
        x.ctrl(qubits[0], qubits[1])

    instance = CustomIntAndListFloat([3], [0, np.pi, 0, np.pi])
    with pytest.raises(RuntimeError) as e:
        counts = cudaq.sample(test, instance)
    assert 'dynamically sized element types for function arguments are not yet supported' in str(
        e.value)
    # Should be: assert len(counts) == 1 and '111' in counts

    # Test that the class can be in a library
    # and the paths all work out
    from mock.hello import TestClass

    @cudaq.kernel
    def test(input: TestClass):
        q = cudaq.qvector(input.i)

    instance = TestClass(2, 2.2)
    state = cudaq.get_state(test, instance)
    state.dump()

    assert len(state) == 2**instance.i

    # Test invalid struct member
    @cudaq.kernel
    def test(input: TestClass):
        local = input.helloBadMember

    with pytest.raises(RuntimeError) as e:
        test.compile()


def test_custom_quantum_type():
    from dataclasses import dataclass

    @dataclass(slots=True)
    class patch:
        data: cudaq.qview
        ancx: cudaq.qview
        ancz: cudaq.qview

    @cudaq.kernel
    def logicalH(p: patch):
        h(p.data)

    # print(logicalH)
    @cudaq.kernel
    def logicalX(p: patch):
        x(p.ancx)

    @cudaq.kernel
    def logicalZ(p: patch):
        z(p.ancz)

    @cudaq.kernel  # (verbose=True)
    def run():
        q = cudaq.qvector(2)
        r = cudaq.qvector(2)
        s = cudaq.qvector(2)
        p = patch(q, r, s)

        logicalH(p)
        logicalX(p)
        logicalZ(p)

    # Test here is that it compiles and runs successfully
    print(run)
    run()


def test_disallow_hybrid_types():
    from dataclasses import dataclass
    # Ensure we don't allow hybrid type s
    @dataclass(slots=True)
    class hybrid:
        q: cudaq.qview
        i: int

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test():
            q = cudaq.qvector(2)
            h = hybrid(q, 1)

        test()

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def testtest(h: hybrid):
            x(h.q[h.i])

        testtest.compile()


def test_disallow_quantum_struct_return():
    from dataclasses import dataclass
    # Ensure we don't allow hybrid type s
    @dataclass(slots=True)
    class T:
        q: cudaq.qview

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test() -> T:
            q = cudaq.qvector(2)
            h = T(q)
            return h

        test()


def test_disallow_recursive_quantum_struct():
    from dataclasses import dataclass

    @dataclass(slots=True)
    class T:
        q: cudaq.qview

    @dataclass(slots=True)
    class Holder:
        t: T

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test():
            q = cudaq.qvector(2)
            t = T(q)
            hh = Holder(t)

        print(test)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test(hh: Holder):
            pass

        print(test)


def test_disallow_struct_with_methods():
    from dataclasses import dataclass

    @dataclass(slots=True)
    class NoCanDo:
        a: cudaq.qview

        def bob(self):
            pass

    try:

        @cudaq.kernel
        def k():
            q = cudaq.qvector(4)
            h = NoCanDo(q)

            print(k())
    except RuntimeError as e:
        assert 'struct types with user specified methods are not allowed.' in str(
            e.value)


def test_issue_9():

    @cudaq.kernel
    def kernel(features: list[float]):
        qubits = cudaq.qvector(8)
        rx(features[0], qubits[100])

    with pytest.raises(RuntimeError) as error:
        kernel([3.14])


def test_issue_1641():

    @cudaq.kernel
    def less_arguments():
        q = cudaq.qubit()
        rx(3.14)

    with pytest.raises(RuntimeError) as error:
        print(less_arguments)
    assert 'missing value' in repr(error)
    assert '(offending source -> rx(3.14))' in repr(error)

    @cudaq.kernel
    def wrong_arguments():
        q = cudaq.qubit()
        rx("random_argument", q)

    with pytest.raises(RuntimeError) as error:
        print(wrong_arguments)
    assert 'cannot convert value' in repr(error)
    assert "(offending source -> rx('random_argument', q))" in repr(error)

    @cudaq.kernel
    def wrong_type():
        q = cudaq.qubit()
        x("random_argument")

    with pytest.raises(RuntimeError) as error:
        print(wrong_type)
    assert 'invalid argument type for target operand' in repr(error)

    @cudaq.kernel
    def invalid_ctrl():
        q = cudaq.qubit()
        rx.ctrl(np.pi, q)

    with pytest.raises(RuntimeError) as error:
        print(invalid_ctrl)
    assert 'missing value' in repr(error)
    assert '(offending source -> rx.ctrl(np.pi, q))' in repr(error)


def test_control_then_adjoint():

    @cudaq.kernel
    def my_func(q: cudaq.qubit, theta: float):
        ry(theta, q)
        rz(theta, q)

    @cudaq.kernel
    def kernel(theta: float):
        ancilla = cudaq.qubit()
        q = cudaq.qubit()

        h(ancilla)
        cudaq.control(my_func, ancilla, q, theta)
        cudaq.adjoint(my_func, q, theta)

    theta = 1.5
    # test here is that this compiles and runs
    cudaq.sample(kernel, theta).dump()


def test_numpy_functions():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        h(q)
        rx(np.pi, q[0])
        ry(np.e, q[1])
        rz(np.euler_gamma, q[2])

    # test here is that this compiles and runs
    cudaq.sample(kernel).dump()

    @cudaq.kernel
    def valid_unsupported():
        q = cudaq.qubit()
        h(q)
        r1(np.inf, q)

    with pytest.raises(RuntimeError):
        cudaq.sample(valid_unsupported)

    @cudaq.kernel
    def invalid_unsupported():
        q = cudaq.qubit()
        h(q)
        r1(np.foo, q)

    with pytest.raises(RuntimeError):
        cudaq.sample(invalid_unsupported)


def test_in_comparator():

    @cudaq.kernel
    def kernel(ind: int):
        q = cudaq.qubit()
        if ind in [6, 13, 20, 27, 34]:
            x(q)

    c = cudaq.sample(kernel, 1)
    assert len(c) == 1 and '0' in c
    c = cudaq.sample(kernel, 20)
    assert len(c) == 1 and '1' in c
    c = cudaq.sample(kernel, 14)
    assert len(c) == 1 and '0' in c
    c = cudaq.sample(kernel, 13)
    assert len(c) == 1 and '1' in c
    c = cudaq.sample(kernel, 26)
    assert len(c) == 1 and '0' in c
    c = cudaq.sample(kernel, 27)
    assert len(c) == 1 and '1' in c
    c = cudaq.sample(kernel, 34)
    assert len(c) == 1 and '1' in c
    c = cudaq.sample(kernel, 36)
    assert len(c) == 1 and '0' in c

    @cudaq.kernel
    def kernel(ind: int):
        q = cudaq.qubit()
        if ind not in [6, 13, 20, 27, 34]:
            x(q)

    c = cudaq.sample(kernel, 1)
    assert len(c) == 1 and '1' in c
    c = cudaq.sample(kernel, 20)
    assert len(c) == 1 and '0' in c


def test_negative_indices_for_a_slice():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(6)
        h(qubits[0:3])
        controls = qubits[0:-1]
        target = qubits[-1]

        x.ctrl(controls, target)

    # test here is that it compiles and runs
    cudaq.sample(kernel)
    circuit = cudaq.draw(kernel)
    print(circuit)
    expected_str = '''     ╭───╮     
q0 : ┤ h ├──●──
     ├───┤  │  
q1 : ┤ h ├──●──
     ├───┤  │  
q2 : ┤ h ├──●──
     ╰───╯  │  
q3 : ───────●──
            │  
q4 : ───────●──
          ╭─┴─╮
q5 : ─────┤ x ├
          ╰───╯
'''

    assert circuit == expected_str


def test_attribute_access_on_call_results():
    """Test that attribute access on call results works correctly."""
    from dataclasses import dataclass

    @dataclass(slots=True)
    class M:
        i: int
        j: int

    @cudaq.kernel
    def test_attribute_on_call() -> int:
        """Test accessing attribute on dataclass constructor call result."""
        return M(6, 9).i

    @cudaq.kernel
    def nested_kernel() -> M:
        return M(6, 9)

    @cudaq.kernel
    def test_attribute_on_kernel_call() -> int:
        return nested_kernel().i

    @cudaq.kernel
    def test_attribute_on_call_with_qubits() -> int:
        """Test accessing attribute on dataclass constructor call result with qubits."""
        q = cudaq.qvector(5)
        return M(6, 9).i

    @cudaq.kernel
    def test_attribute_on_call_in_assignment() -> int:
        """Test accessing attribute on dataclass constructor call result in assignment."""
        foo = M(6, 9).i
        return foo

    @cudaq.kernel
    def test_attribute_on_call_in_expression() -> int:
        """Test accessing attribute on dataclass constructor call result in expression."""
        return M(6, 9).i + M(3, 4).j

    # Test that all kernels compile and run successfully
    result1 = cudaq.run(test_attribute_on_call)[0]
    assert result1 == 6

    result2 = cudaq.run(test_attribute_on_kernel_call)[0]
    assert result2 == 6

    result3 = cudaq.run(test_attribute_on_call_with_qubits)[0]
    assert result3 == 6

    result4 = cudaq.run(test_attribute_on_call_in_assignment)[0]
    assert result4 == 6

    result5 = cudaq.run(test_attribute_on_call_in_expression)[0]
    assert result5 == 10  # 6 + 4

    # Test with different dataclass
    @dataclass(slots=True)
    class Point:
        x: float
        y: float

    @cudaq.kernel
    def test_point_attribute() -> float:
        """Test accessing attribute on different dataclass."""
        return Point(3.14, 2.71).x

    result5 = cudaq.run(test_point_attribute)[0]
    assert abs(result5 - 3.14) < 1e-6


def test_mid_circuit_measurements():

    @cudaq.kernel
    def callee(register: cudaq.qview):
        for i in range(4):
            if i % 2 == 0:
                x(register[i])

            m = mz(register[i])
            reset(register[i])

            if m:
                x(register[i])
            else:
                h(register[i])

    @cudaq.kernel
    def caller():
        qr = cudaq.qvector(4)
        callee(qr)

    counts = cudaq.sample(caller)
    assert counts.register_names == ["__global__", "m"]

    globalCounts = counts.get_register_counts("__global__")
    assert len(globalCounts) == 4
    assert "1010" in globalCounts
    assert "1011" in globalCounts
    assert "1110" in globalCounts
    assert "1111" in globalCounts

    mCounts = counts.get_register_counts("m")
    assert len(mCounts) == 1
    assert "1010" in mCounts


def test_error_on_non_callable_type():

    @cudaq.kernel
    def kernel(op: cudaq.pauli_word):
        q = cudaq.qvector(2)
        x(q[1])
        op(q[0])

    with pytest.raises(RuntimeError) as e:
        result = cudaq.sample(kernel, cudaq.pauli_word("X"))
    assert "object is not callable" in str(e.value)


# leave for gdb debugging
if __name__ == "__main__":
    test_custom_classical_kernel_type()
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
