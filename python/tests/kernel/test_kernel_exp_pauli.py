# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import math

import cudaq
import pytest


def test_exp_pauli():

    @cudaq.kernel
    def test():
        q = cudaq.qvector(2)
        exp_pauli(1.0, q, "XX")

    counts = cudaq.sample(test)
    assert '00' in counts
    assert '11' in counts
    assert not '01' in counts
    assert not '10' in counts


def test_exp_pauli_param():

    @cudaq.kernel
    def test_param(w: cudaq.pauli_word):
        q = cudaq.qvector(2)
        exp_pauli(1.0, q, w)

    counts = cudaq.sample(test_param, cudaq.pauli_word("XX"))
    assert '00' in counts
    assert '11' in counts
    assert not '01' in counts
    assert not '10' in counts


def test_exp_pauli_individual_qubits():

    @cudaq.kernel
    def test(theta: float):
        q = cudaq.qvector(3)
        exp_pauli(theta, "YX", q[0], q[2])

    counts = cudaq.sample(test, math.pi / 2.0)
    assert len(counts) == 1
    assert "101" in counts


def test_exp_pauli_single_individual_qubit():

    @cudaq.kernel
    def test(theta: float):
        q = cudaq.qvector(3)
        exp_pauli(theta, "X", q[1])

    counts = cudaq.sample(test, math.pi / 2.0)
    assert len(counts) == 1
    assert "010" in counts


def test_exp_pauli_three_individual_qubits():

    @cudaq.kernel
    def test(theta: float):
        q = cudaq.qvector(3)
        exp_pauli(theta, "XXX", q[2], q[0], q[1])

    counts = cudaq.sample(test, math.pi / 2.0)
    assert len(counts) == 1
    assert "111" in counts


def test_exp_pauli_individual_qubit_ordering():

    @cudaq.kernel
    def test(theta: float):
        q = cudaq.qvector(2)
        # Ordering-sensitive: X flips q[0], while Z leaves q[1] in |0>.
        exp_pauli(theta, "XZ", q[0], q[1])

    counts = cudaq.sample(test, math.pi / 2.0)
    assert len(counts) == 1
    assert "10" in counts


def test_exp_pauli_single_individual_target_must_be_qubit():

    @cudaq.kernel
    def test():
        exp_pauli(1.0, "X", 0)

    with pytest.raises(RuntimeError,
                       match="invalid argument type for target operand"):
        test.compile()


def test_exp_pauli_variadic_targets_must_be_qubits():

    @cudaq.kernel
    def test():
        q = cudaq.qvector(1)
        exp_pauli(1.0, "XX", q[0], 0)

    with pytest.raises(RuntimeError,
                       match="invalid argument type for target operand"):
        test.compile()
