# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
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

## [PYTHON_VERSION_FIX]
skipIfPythonLessThan39 = pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="built-in collection types such as `list` not supported")


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


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


def test_dynamic_circuit():
    """Test that we correctly sample circuits with 
       mid-circuit measurements and conditionals."""

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


@skipIfPythonLessThan39
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
    def qernel(a: float, b: float) -> float:
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

    print(kernel3)
    counts = cudaq.sample(kernel3, 5)
    assert len(counts) == 1
    assert '1' * 5 in counts

    @cudaq.kernel
    def kernel4(myList: List[int]):
        q = cudaq.qvector(len(myList))
        casted = list(myList)
        for i in casted:
            x(q[i])

    print(kernel4)
    counts = cudaq.sample(kernel4, list(range(5)))
    assert len(counts) == 1
    assert '1' * 5 in counts


@skipIfPythonLessThan39
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


@skipIfPythonLessThan39
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


def test_capture_disallow_change_variable():

    n = 3

    @cudaq.kernel
    def kernel() -> int:
        if True:
            cudaq.dbg.ast.print_i64(n)
            # Change n, emits an error
            n = 4
        return n

    with pytest.raises(RuntimeError) as e:
        kernel()


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
