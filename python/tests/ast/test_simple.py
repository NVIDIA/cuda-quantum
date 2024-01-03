# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import numpy as np

import cudaq


@pytest.fixture(autouse=True)
def do_something():
    cudaq.__clearKernelRegistries()
    yield
    return


def test_bell():

    @cudaq.kernel(jit=True)
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])

    print(bell)
    bell()
    counts = cudaq.sample(bell)
    assert '00' in counts and '11' in counts


def test_ghz():

    @cudaq.kernel(jit=True, verbose=True)
    def ghz(N: int):
        q = cudaq.qvector(N)
        h(q[0])
        for i in range(N - 1):
            x.ctrl(q[i], q[i + 1])

    print(ghz)

    counts = cudaq.sample(ghz, 5)
    assert '0' * 5 in counts and '1' * 5 in counts


def test_no_annotations():
    with pytest.raises(RuntimeError) as error:

        @cudaq.kernel(jit=True, verbose=True)
        def ghz(N):
            q = cudaq.qvector(N)
            h(q[0])
            for i in range(N - 1):
                x.ctrl(q[i], q[i + 1])


def test_kernel_composition():

    @cudaq.kernel(jit=True, verbose=True)
    def iqft(qubits: cudaq.qview):
        N = qubits.size()
        for i in range(N // 2):
            swap(qubits[i], qubits[N - i - 1])

        for i in range(N - 1):
            h(qubits[i])
            j = i + 1
            for y in range(i, -1, -1):
                r1.ctrl(-np.pi / 2**(j - y), qubits[j], qubits[y])

        h(qubits[N - 1])

    print(iqft)

    @cudaq.kernel(jit=True, verbose=True)
    def entryPoint():
        q = cudaq.qvector(3)
        iqft(q)

    print(entryPoint)
    entryPoint()


def test_qreg_iter():

    @cudaq.kernel(jit=True, verbose=True)
    def foo(N: int):
        q = cudaq.qvector(N)
        for r in q:
            x(r)

    foo(10)


def test_control_kernel():

    @cudaq.kernel(jit=True)
    def applyX(q: cudaq.qubit):
        x(q)

    @cudaq.kernel(jit=True, verbose=True)
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        cudaq.control(applyX, [q[0]], q[1])
        cudaq.control(applyX, q[0], q[1])
        cudaq.control(applyX, q[0], q[1])

    print(bell)
    bell()


def test_simple_sampling_qpe():
    """Test that we can build up a set of kernels, compose them, and sample."""

    @cudaq.kernel(jit=True)
    def iqft(qubits: cudaq.qview):
        N = qubits.size()
        for i in range(N // 2):
            swap(qubits[i], qubits[N - i - 1])

        for i in range(N - 1):
            h(qubits[i])
            j = i + 1
            for k, y in enumerate(range(i, -1, -1)):
                r1.ctrl(-np.pi / 2**(j - y), qubits[j], qubits[y])

        h(qubits[N - 1])

    @cudaq.kernel(jit=True)
    def tGate(qubit: cudaq.qubit):
        t(qubit)

    @cudaq.kernel(jit=True)
    def xGate(qubit: cudaq.qubit):
        x(qubit)

    @cudaq.kernel(jit=True, verbose=True)
    def qpe(nC: int, nQ: int):
        q = cudaq.qvector(nC + nQ)
        countingQubits = q.front(nC)
        stateRegister = q.back()
        xGate(stateRegister)
        h(countingQubits)
        for i in range(nC):
            for j in range(2**i):
                cudaq.control(tGate, [countingQubits[i]], stateRegister)
        iqft(countingQubits)
        # mz(countingQubits)

    print(qpe)
    qpe(3, 1)


def test_enumerate():

    @cudaq.kernel(jit=True, verbose=True)
    def simple(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i, qubit in enumerate(qubits.front(numQubits - 1)):
            x.ctrl(qubit, qubits[i + 1])

    cudaq.sample(simple, 5).dump()

    @cudaq.kernel(jit=True, verbose=True)
    def simple2(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i, qubitIdx in enumerate(range(numQubits - 1)):
            x.ctrl(qubits[i], qubits[qubitIdx + 1])

    simple2(3)


def test_adjoint():
    """Test that adjoint can be called on kernels and operations."""

    @cudaq.kernel(jit=True, verbose=True)
    def single_adjoint_test():
        q = cudaq.qubit()
        t(q)
        t.adj(q)

    counts = cudaq.sample(single_adjoint_test)
    assert '0' in counts
    assert len(counts) == 1

    @cudaq.kernel(jit=True, verbose=True)
    def qvector_adjoint_test():
        q = cudaq.qvector(2)
        t(q)
        t.adj(q)

    counts = cudaq.sample(qvector_adjoint_test)
    assert '00' in counts
    assert len(counts) == 1

    @cudaq.kernel(jit=True, verbose=True)
    def rotation_adjoint_test():
        q = cudaq.qubit()
        rx(1.1, q)
        rx.adj(1.1, q)

        ry(1.1, q)
        ry.adj(1.1, q)

    counts = cudaq.sample(rotation_adjoint_test)
    assert '0' in counts
    assert len(counts) == 1

    @cudaq.kernel(jit=True, verbose=True)
    def test_kernel_adjoint(q: cudaq.qview):
        h(q[0])
        t(q[1])
        s(q[2])

    @cudaq.kernel(jit=True, verbose=True)
    def test_caller():
        q = cudaq.qvector(3)
        x(q[0])
        x(q[2])
        test_kernel_adjoint(q)
        cudaq.adjoint(test_kernel_adjoint, q)

    counts = cudaq.sample(test_caller)
    assert len(counts) == 1
    assert '101' in counts


def test_synth_and_qir():

    @cudaq.kernel(jit=True, verbose=True)
    def ghz(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i, qubitIdx in enumerate(range(numQubits - 1)):
            x.ctrl(qubits[i], qubits[qubitIdx + 1])

    print(ghz)
    ghz_synth = cudaq.synthesize(ghz, 5)
    print(ghz_synth)
    print(cudaq.to_qir(ghz_synth))
    print(cudaq.to_qir(ghz_synth, profile='qir-base'))
    ghz_synth()


from typing import Callable


def test_callCallable():

    @cudaq.kernel(jit=True)
    def xGate(q: cudaq.qubit):
        x(q)

    print(xGate)

    @cudaq.kernel(jit=True)  #, verbose=True)
    def callXGate(functor: Callable[[cudaq.qubit], None]):
        q = cudaq.qvector(2)
        functor(q[0])

    print(callXGate)
    callXGate(xGate)


# TODO if stmts, while loop,
#  async, exp_pauli, common kernels, kernel function parameter
