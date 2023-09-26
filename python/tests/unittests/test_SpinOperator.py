# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest

import collections

import cudaq
from cudaq import spin
import numpy as np


def assert_close(want, got, tolerance=1.e-5) -> bool:
    return abs(want - got) < tolerance


def test_spin_class():
    """
    Tests that we can construct each of the convenience functions for 
    the Pauli spin operators on different qubits.
    """
    qubit = 0
    i_ = spin.i(target=qubit)
    x_ = spin.x(target=qubit)
    y_ = spin.y(qubit)
    z_ = spin.z(qubit)

    data, coeffs = i_.get_raw_data()
    assert (len(data) == 1)
    assert (len(data[0]) == 2)
    assert (data[0] == [0, 0])

    data, coeffs = x_.get_raw_data()
    assert (len(data) == 1)
    assert (len(data[0]) == 2)
    assert (data[0] == [1, 0])

    data, coeffs = y_.get_raw_data()
    assert (len(data) == 1)
    assert (len(data[0]) == 2)
    assert (data[0] == [1, 1])

    data, coeffs = z_.get_raw_data()
    assert (len(data) == 1)
    assert (len(data[0]) == 2)
    assert (data[0] == [0, 1])


def test_spin_op_operators():
    """
    Tests the binary operators on the `cudaq.SpinOperator` class. We're just 
    testing that each runs without error and constructs two example strings 
    that we can verify against. We are not fully testing the accuracy of 
    each individual operator at the moment.
    """
    # Test the empty (identity) constructor.
    spin_a = cudaq.SpinOperator()
    spin_b = spin.x(0)
    # Test the copy constructor.
    spin_b_copy = cudaq.SpinOperator(spin_operator=spin_b)
    assert (spin_b_copy == spin_b)
    assert (spin_b_copy != spin_a)
    spin_c = spin.y(1)
    spin_d = spin.z(2)

    print("Start", spin_a)

    # In-place operators:
    # this += SpinOperator
    spin_a += spin_b
    print('next ', spin_a)
    # this -= SpinOperator
    spin_a -= spin_c

    # this *= SpinOperator
    spin_a *= spin_d
    # this *= float/double
    spin_a *= 5.0
    # this *= complex
    spin_a *= (1.0 + 1.0j)

    # Other operators:
    # SpinOperator + SpinOperator
    spin_f = spin_a + spin_b
    # SpinOperator - SpinOperator
    spin_g = spin_a - spin_b
    # SpinOperator * SpinOperator
    spin_h = spin_a * spin_b
    # SpinOperator * double
    spin_i = spin_a * -1.0
    # double * SpinOperator
    spin_j = -1.0 * spin_a
    # SpinOperator * complex
    spin_k = spin_a * (1.0 + 1.0j)
    # complex * SpinOperator
    spin_l = (1.0 + 1.0j) * spin_a
    # SpinOperator + double
    spin_m = spin_a + 3.0
    # double + SpinOperator
    spin_n = 3.0 + spin_a
    # SpinOperator - double
    spin_o = spin_a - 3.0
    # double - SpinOperator
    spin_p = 3.0 - spin_a

    data, coeffs = spin_a.get_raw_data()
    assert (len(data) == 3)
    assert (len(data[0]) == 6)
    expected = [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [5 + 5j, 5 + 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_b.get_raw_data()
    assert (len(data) == 1)
    assert (len(data[0]) == 2)
    expected = [[1, 0]]
    assert (all([d in expected for d in data]))
    expected = [1]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_c.get_raw_data()
    assert (len(data) == 1)
    assert (len(data[0]) == 4)
    expected = [[0, 1, 0, 1]]
    assert (all([d in expected for d in data]))
    expected = [1]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_d.get_raw_data()
    assert (len(data) == 1)
    assert (len(data[0]) == 6)
    expected = [[0, 0, 0, 0, 0, 1]]
    assert (all([d in expected for d in data]))
    expected = [1]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_f.get_raw_data()
    assert (len(data) == 4)
    assert (len(data[0]) == 6)
    expected = [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 0]]
    assert (all([d in expected for d in data]))
    expected = [5 + 5j, 5 + 5j, -5 - 5j, 1]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_g.get_raw_data()
    assert (len(data) == 4)
    assert (len(data[0]) == 6)
    expected = [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 0]]
    assert (all([d in expected for d in data]))
    expected = [5 + 5j, 5 + 5j, -5 - 5j, -1]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_h.get_raw_data()
    assert (len(data) == 3)
    assert (len(data[0]) == 6)
    expected = [[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [5 + 5j, 5 + 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_i.get_raw_data()
    assert (len(data) == 3)
    assert (len(data[0]) == 6)
    expected = [[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [-5 - 5j, 5 + 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_j.get_raw_data()
    assert (len(data) == 3)
    assert (len(data[0]) == 6)
    expected = [[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [-5 - 5j, 5 + 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_k.get_raw_data()
    assert (len(data) == 3)
    assert (len(data[0]) == 6)
    expected = [[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [10j, 10j, -10j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_l.get_raw_data()
    assert (len(data) == 3)
    assert (len(data[0]) == 6)
    expected = [[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [10j, 10j, -10j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_m.get_raw_data()
    assert (len(data) == 4)
    assert (len(data[0]) == 6)
    expected = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [3, 5 + 5j, 5 + 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_o.get_raw_data()
    assert (len(data) == 4)
    assert (len(data[0]) == 6)
    expected = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [-3, 5 + 5j, 5 + 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_p.get_raw_data()
    assert (len(data) == 4)
    assert (len(data[0]) == 6)
    expected = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [3, 5 + 5j, -5 - 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))


def test_spin_op_members():
    """
    Test all of the bound member functions on the `cudaq.SpinOperator` class.
    """
    spin_operator = cudaq.SpinOperator()
    # Assert that it's the identity.
    assert spin_operator.is_identity()
    # Only have 1 term and 1 qubit.
    assert spin_operator.get_term_count() == 1
    assert spin_operator.get_qubit_count() == 1
    spin_operator += -1.0 * spin.x(1)
    # Should now have 2 terms and 2 qubits.
    assert spin_operator.get_term_count() == 2
    assert spin_operator.get_qubit_count() == 2
    # No longer identity.
    assert not spin_operator.is_identity()
    for term in spin_operator:
        # Second term should have a coefficient of -1.0
        assert term.get_coefficient() == -1.0 or term.get_coefficient() == 1.0
        assert term.get_coefficient() == -1.0 or term.get_coefficient() == 1.0


def test_spin_op_vqe():
    """
    Test the `cudaq.SpinOperator` class on a simple VQE Hamiltonian.
    """
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    print(hamiltonian)
    # Checking equality operators.
    assert spin.x(2) != hamiltonian
    assert hamiltonian == hamiltonian
    assert hamiltonian.get_term_count() == 5

    got_data, got_coefficients = hamiltonian.get_raw_data()
    assert (len(got_data) == 5)
    assert (len(got_data[0]) == 4)
    expected = [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1],
                [0, 0, 1, 0]]
    assert (all([d in expected for d in got_data]))
    expected = [5.907, -2.1433, -2.1433, .21829, -6.125]
    assert (all([c in expected for c in got_coefficients]))


def test_matrix():
    """
    Test that the `cudaq.SpinOperator` can produce its matrix representation 
    and that we can use that matrix with standard python packages like numpy.
    """
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    mat = hamiltonian.to_matrix()
    assert assert_close(-1.74, np.linalg.eigvals(mat)[0], 1e-2)
    print(mat)
    want_matrix = np.array([[.00029, 0, 0, 0], [0, 12.2503, -4.2866, 0],
                            [0, -4.2866, -.43629, 0], [0, 0, 0, 11.8137]])

    got_matrix = np.array(mat, copy=False)
    assert np.allclose(want_matrix, got_matrix, rtol=1e-3)


def test_spin_op_foreach():
    """
    Test the `cudaq.SpinOperator` for_each_term method
    """
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    print(hamiltonian)

    counter = 0

    def doSomethingWithTerm(term):
        nonlocal counter
        print(term)
        counter += 1

    hamiltonian.for_each_term(doSomethingWithTerm)

    assert counter == 5

    counter = 0
    xSupports = []

    def doSomethingWithTerm(term):

        def doSomethingWithPauli(pauli: cudaq.Pauli, idx: int):
            nonlocal counter, xSupports
            if pauli == cudaq.Pauli.X:
                counter = counter + 1
                xSupports.append(idx)

        term.for_each_pauli(doSomethingWithPauli)

    hamiltonian.for_each_term(doSomethingWithTerm)

    assert counter == 2
    assert xSupports == [0, 1]


def test_spin_op_iter():
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    count = 0
    for term in hamiltonian:
        count += 1
    assert count == 5


def test_spin_op_sparse_matrix():
    """
    Test that the `cudaq.SpinOperator` can produce its sparse matrix representation 
    and that we can use that matrix with standard python packages like numpy.
    """
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    numQubits = hamiltonian.get_qubit_count()
    mat = hamiltonian.to_matrix()
    data, rows, cols = hamiltonian.to_sparse_matrix()
    for i, value in enumerate(data):
        print(rows[i], cols[i], value)
        assert np.isclose(mat[rows[i], cols[i]], value)

    # can use scipy
    # scipyM = scipy.sparse.csr_array((data, (rows, cols)), shape=(2**numQubits,2**numQubits))
    # E, ev = scipy.sparse.linalg.eigsh(scipyM, k=1, which='SA')
    # assert np.isclose(E[0], -1.7488, 1e-2)


def test_spin_op_from_word():
    s = cudaq.SpinOperator.from_word("ZZZ")
    want_spin_op = spin.z(0) * spin.z(1) * spin.z(2)
    got_spin_op = cudaq.SpinOperator.from_word("ZZZ")
    assert got_spin_op == want_spin_op

    s = cudaq.SpinOperator.from_word("XXIXYZ")
    want_spin_op = spin.x(0) * spin.x(1) * spin.i(2) * spin.x(3) * spin.y(
        4) * spin.z(5)
    got_spin_op = cudaq.SpinOperator.from_word("XXIXYZ")
    assert got_spin_op == want_spin_op

    want_spin_op = spin.x(0) * spin.y(1) * spin.z(2)
    got_spin_op = cudaq.SpinOperator.from_word("XYZ")
    assert got_spin_op == want_spin_op


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
