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
from cudaq import spin, x, z, h, ry, t, swap, r1, mz, rx, s


def test_simple_sampling_ghz():
    """Test that we can build a very simple kernel and sample it."""
    @cudaq.kernel
    def simple(numQubits):
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i, qubit in enumerate(qubits.front(numQubits-1)):
            x.ctrl(qubit, qubits[i+1])

    counts = cudaq.sample(simple, 10)
    assert len(counts) == 2
    assert '0'*10 in counts and '1'*10 in counts


def test_simple_sampling_qpe():
    """Test that we can build up a set of kernels, compose them, and sample."""
    @cudaq.kernel
    def iqft(qubits):
        N = qubits.size()
        for i in range(N//2):
            swap(qubits[i], qubits[N-i-1])

        for i in range(N-1):
            h(qubits[i])
            j = i + 1
            for y in range(i, -1, -1):
                r1.ctrl(-np.pi / 2**(j-y), qubits[j], qubits[y])

        h(qubits[N-1])

    @cudaq.kernel
    def tGate(qubit):
        t(qubit)

    @cudaq.kernel
    def xGate(qubit):
        x(qubit)

    @cudaq.kernel
    def qpe(nC, nQ, statePrep, oracle):
        q = cudaq.qvector(nC+nQ)
        countingQubits = q.front(nC)
        stateRegister = q.back()
        statePrep(stateRegister)
        h(countingQubits)
        for i in range(nC):
            for j in range(2**i):
                cudaq.control(oracle, [countingQubits[i]], stateRegister)
        iqft(countingQubits)
        mz(countingQubits)

    cudaq.set_random_seed(13)
    counts = cudaq.sample(qpe, 3, 1, xGate, tGate)
    assert len(counts) == 1
    assert '100' in counts


def test_broadcast():
    """Test that sample and observe broadcasting works."""

    @cudaq.kernel
    def circuit(inSize: int):
        qubits = cudaq.qvector(inSize)
        h(qubits[0])
        for i in range(inSize-1):
            x.ctrl(qubits[i], qubits[i+1])
    
    cudaq.set_random_seed(13)
    allCounts = cudaq.sample(circuit, [3, 4, 5, 6, 7])
    first0 = '000'
    first1 = '111'
    for c in allCounts:
        print(c)
        assert first0 in c and first1 in c
        first0 += '0'
        first1 += '1'

    np.random.seed(13)
    testNpArray = np.random.randint(3, high=8, size=6)
    print(testNpArray)
    allCounts = cudaq.sample(circuit, testNpArray)
    for i, c in enumerate(allCounts):
        print(c)
        assert '0' * testNpArray[i] in c and '1' * testNpArray[i] in c

    @cudaq.kernel
    def circuit(angles: list):
        q = cudaq.qvector(2)
        rx(angles[0], q[0])
        ry(angles[1], q[0])
        x.ctrl(q[0], q[1])

    runtimeAngles = np.random.uniform(low=1.0, high=np.pi, size=(10, 2))
    print(runtimeAngles)
    allCounts = cudaq.sample(circuit, runtimeAngles)
    for i, c in enumerate(allCounts):
        print(runtimeAngles[i, :], c)
        assert len(c) == 2
