# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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


def test_list_bool():

    @cudaq.kernel
    def kernel5(myList: list[bool]):
        q = cudaq.qvector(len(myList))
        for i, b in enumerate(myList):
            if b:
                x(q[i])

    counts = cudaq.sample(kernel5, [True, False, True, False])
    assert len(counts) == 1
    assert '1010' in counts


def test_list_pauli():

    @cudaq.kernel
    def kernel(n: int, words: list[cudaq.pauli_word], theta: float):
        qubits = cudaq.qvector(n)
        for term in words:
            exp_pauli(theta, qubits, term)

    words = [cudaq.pauli_word('YZXI'), cudaq.pauli_word('XZYI')]
    counts = cudaq.sample(kernel, 4, words, 0.5)
    assert len(counts) == 2
    assert '0000' in counts
    assert '1010' in counts


def test_list_pauli_str():

    @cudaq.kernel
    def kernel(n: int, words: list[cudaq.pauli_word], theta: float):

        qubits = cudaq.qvector(n)
        for term in words:
            exp_pauli(theta, qubits, term)

    words = ['YZXI', 'XZYI']
    counts = cudaq.sample(kernel, 4, words, 0.5)
    assert len(counts) == 2
    assert '0000' in counts
    assert '1010' in counts


def test_nested_list_bool():

    @cudaq.kernel
    def kernel5(n: int, myList: list[list[bool]]):
        q = cudaq.qvector(n)
        for i, inner in enumerate(myList):
            for j, e in enumerate(inner):
                m = i * len(inner) + j
                cudaq.dbg.ast.print_i64(m)
                if e:
                    x(q[m])
                    cudaq.dbg.ast.print_i64(1)
                else:
                    cudaq.dbg.ast.print_i64(0)

    counts = cudaq.sample(
        kernel5, 8, [[True, False, True, False], [False, True, False, True]])
    print(counts)
    assert len(counts) == 1
    assert '10100101' in counts


def test_nested_list_int():

    @cudaq.kernel
    def kernel6(n: int, myList: list[list[int]]):
        q = cudaq.qvector(n)
        for inner in myList:
            for i in inner:
                x(q[i])

    counts = cudaq.sample(kernel6, 8, [[0, 1, 2, 3], [4, 5, 6, 7]])
    assert len(counts) == 1
    assert '11111111' in counts


def test_nested_list_float():

    @cudaq.kernel
    def kernel7(n: int, myList: list[list[float]]):
        q = cudaq.qvector(n)
        for inner in myList:
            for i in inner:
                j = int(i)
                x(q[j])

    counts = cudaq.sample(kernel7, 8, [[0., 1., 2., 3.], [4., 5., 6., 7.]])
    assert len(counts) == 1
    assert '11111111' in counts


def test_nested_list_pauli():

    @cudaq.kernel
    def kernel(qubit_num: int, words: list[list[cudaq.pauli_word]],
               theta: list[float]):

        qubits = cudaq.qvector(qubit_num)
        for i in range(len(words)):
            for term in words[i]:
                exp_pauli(theta[i], qubits, term)

    words = [[cudaq.pauli_word('YZXI'),
              cudaq.pauli_word('XZYI')],
             [cudaq.pauli_word('IYZX'),
              cudaq.pauli_word('IXZY')]]
    counts = cudaq.sample(kernel, 4, words, [0.5, 0.5])
    assert len(counts) == 4
    assert '0000' in counts
    assert '0101' in counts
    assert '1010' in counts
    assert '1111' in counts


def test_nested_list_pauli_str():

    @cudaq.kernel
    def kernel(qubit_num: int, words: list[list[cudaq.pauli_word]],
               theta: list[float]):

        qubits = cudaq.qvector(qubit_num)
        for i in range(len(words)):
            for term in words[i]:
                exp_pauli(theta[i], qubits, term)

    words = [['YZXI', 'XZYI'], ['IYZX', 'IXZY']]
    counts = cudaq.sample(kernel, 4, words, [0.5, 0.5])
    assert len(counts) == 4
    assert '0000' in counts
    assert '0101' in counts
    assert '1010' in counts
    assert '1111' in counts


def test_nested_list3_bool():

    @cudaq.kernel
    def kernel5(n: int, myList: list[list[list[bool]]]):
        q = cudaq.qvector(n)
        for k, inner in enumerate(myList):
            for i, inner2 in enumerate(inner):
                for j, e in enumerate(inner2):
                    m = k * len(inner) * len(inner2) + i * len(inner2) + j
                    cudaq.dbg.ast.print_i64(m)
                    if e:
                        x(q[m])
                        cudaq.dbg.ast.print_i64(1)
                    else:
                        cudaq.dbg.ast.print_i64(0)

    counts = cudaq.sample(
        kernel5, 16, [[[True, False, True, False], [False, True, False, True]],
                      [[True, False, True, False], [False, True, False, True]]])
    print(counts)
    assert len(counts) == 1
    assert '1010010110100101' in counts


def test_nested_list3_int():

    @cudaq.kernel
    def kernel6(n: int, myList: list[list[list[int]]]):
        q = cudaq.qvector(n)
        for inner in myList:
            for inner2 in inner:
                for i in inner2:
                    x(q[i])

    counts = cudaq.sample(
        kernel6, 16,
        [[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]])
    assert len(counts) == 1
    assert '1111111111111111' in counts


def test_nested_list4_int():

    @cudaq.kernel
    def kernel6(n: int, myList: list[list[list[list[int]]]]):
        q = cudaq.qvector(n)
        for inner in myList:
            for inner2 in inner:
                for inner3 in inner2:
                    for i in inner3:
                        cudaq.dbg.ast.print_i64(i)
                        x(q[i])

    counts = cudaq.sample(kernel6, 16,
                          [[[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                           [[[8, 9], [10, 11]], [[12, 13], [14, 15]]]])
    assert len(counts) == 1
    print(counts)
    assert '1111111111111111' in counts


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
