# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest

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

    assert (str(i_) == "(1,0) I" + str(qubit))
    assert (str(x_) == "(1,0) X" + str(qubit))
    assert (str(y_) == "(1,0) Y" + str(qubit))
    assert (str(z_) == "(1,0) Z" + str(qubit))


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

    # In-place operators:
    # this += SpinOperator
    spin_a += spin_b
    # this -= SpinOperator
    spin_a -= spin_c
    # this *= SpinOperator
    spin_a *= spin_d
    # this *= float/double
    spin_a *= 5.0
    # this *= complex
    spin_a *= (1.0 + 1.0j)

    # Other operators:
    # this[idx]
    spin_e = spin_a[1]
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

    assert (str(spin_a) == "(5,5) I0I1Z2 + (5,5) X0I1Z2 + (-5,-5) I0Y1Z2")
    assert (str(spin_b) == "(1,0) X0")
    assert (str(spin_c) == "(1,0) I0Y1")
    assert (str(spin_d) == "(1,0) I0I1Z2")
    assert (str(spin_e) == "(5,5) X0I1Z2")
    assert (str(spin_f) ==
            "(5,5) I0I1Z2 + (5,5) X0I1Z2 + (-5,-5) I0Y1Z2 + (1,0) X0I1I2")
    assert (str(spin_g) ==
            "(5,5) I0I1Z2 + (5,5) X0I1Z2 + (-5,-5) I0Y1Z2 + (-1,-0) X0I1I2")
    assert (str(spin_h) == "(5,5) X0I1Z2 + (5,5) I0I1Z2 + (-5,-5) X0Y1Z2")
    assert (str(spin_i) == "(-5,-5) I0I1Z2 + (-5,-5) X0I1Z2 + (5,5) I0Y1Z2")
    assert (str(spin_j) == "(-5,-5) I0I1Z2 + (-5,-5) X0I1Z2 + (5,5) I0Y1Z2")
    assert (str(spin_k) == "(0,10) I0I1Z2 + (0,10) X0I1Z2 + (0,-10) I0Y1Z2")
    assert (str(spin_l) == "(0,10) I0I1Z2 + (0,10) X0I1Z2 + (0,-10) I0Y1Z2")
    assert (str(spin_m) ==
            "(3,0) I0I1I2 + (5,5) I0I1Z2 + (5,5) X0I1Z2 + (-5,-5) I0Y1Z2")
    assert (str(spin_o) ==
            "(5,5) I0I1Z2 + (5,5) X0I1Z2 + (-5,-5) I0Y1Z2 + (-3,-0) I0I1I2")
    assert (str(spin_p) ==
            "(3,0) I0I1I2 + (-5,-5) I0I1Z2 + (-5,-5) X0I1Z2 + (5,5) I0Y1Z2")


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
    # Second term should have a coefficient of -1.0
    assert spin_operator.get_term_coefficient(1) == -1.0
    # assert spin_operator.get_coefficients() ==
    want_string = "(1,0) I0I1 + (-1,-0) I0X1"
    # Check that we're converting to the proper string.
    assert spin_operator.to_string() == want_string
    assert str(spin_operator) == want_string
    # Print to terminal.
    spin_operator.dump()


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
    # Checking the indexing operator.
    assert hamiltonian[1].to_string() == "(-2.1433,-0) X0X1"
    assert hamiltonian.get_term_count() == 5
    want_string = "(5.907,0) I0I1 + (-2.1433,-0) X0X1 + (-2.1433,-0) Y0Y1 + (0.21829,0) Z0I1 + (-6.125,-0) I0Z1"
    assert want_string == hamiltonian.to_string()
    assert want_string == str(hamiltonian)

def test_matrix():
    """
    Test that the `cudaq.SpinOperator` can produce its matrix representation 
    and that we can use that matrix with standard python packages like numpy.
    """
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    mat = hamiltonian.to_matrix()
    assert_close(-1.74, np.linalg.eigvals(mat)[0], 1e-2)

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
        counter = counter+ 1
    
    hamiltonian.for_each_term(doSomethingWithTerm)

    assert counter == 5

    counter = 0
    xSupports = []
    def doSomethingWithTerm(term):
        def doSomethingWithPauli(pauli : cudaq.Pauli, idx: int):
            nonlocal counter, xSupports
            if pauli == cudaq.Pauli.X:
                counter = counter+1
                xSupports.append(idx)
        term.for_each_pauli(doSomethingWithPauli)
    
    hamiltonian.for_each_term(doSomethingWithTerm)

    assert counter == 2
    assert xSupports == [0,1]



# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
