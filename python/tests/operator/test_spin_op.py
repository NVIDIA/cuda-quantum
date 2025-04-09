# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np, os, pytest, random
from cudaq import Pauli
from cudaq.spin_op import *
from cudaq import spin # FIXME: REMOVE
from op_utils import * # test helpers


@pytest.fixture(autouse=True)
def setup():
    random.seed(10)
    yield


def test_definitions():
    assert np.allclose(i(1).to_matrix(), identity_matrix(2))
    assert np.allclose(x(1).to_matrix(), paulix_matrix())
    assert np.allclose(y(1).to_matrix(), pauliy_matrix())
    assert np.allclose(z(1).to_matrix(), pauliz_matrix())
    # FIXME: PLUS AND MINUS

    # legacy test cases

    qubit = 0
    i_ = spin.i(target=qubit)
    x_ = spin.x(target=qubit)
    y_ = spin.y(qubit)
    z_ = spin.z(qubit)

    data, _ = i_.get_raw_data()
    assert (len(data) == 1)
    assert (len(data[0]) == 2)
    assert (data[0] == [0, 0])

    data, _ = x_.get_raw_data()
    assert (len(data) == 1)
    assert (len(data[0]) == 2)
    assert (data[0] == [1, 0])

    data, _ = y_.get_raw_data()
    assert (len(data) == 1)
    assert (len(data[0]) == 2)
    assert (data[0] == [1, 1])

    data, _ = z_.get_raw_data()
    assert (len(data) == 1)
    assert (len(data[0]) == 2)
    assert (data[0] == [0, 1])


def test_construction():
    prod = identity()
    assert np.allclose(prod.to_matrix(), identity_matrix(1))
    prod *= z(0)
    assert np.allclose(prod.to_matrix(), pauliz_matrix())
    sum = empty()
    assert np.allclose(sum.to_matrix(), zero_matrix(1))
    sum *= z(0)
    assert sum.degrees() == []
    assert np.allclose(sum.to_matrix(), zero_matrix(1))
    sum += identity(1)
    assert sum.degrees() == [1]
    assert np.allclose(sum.to_matrix(), identity_matrix(2))
    sum *= z(1)
    assert np.allclose(sum.to_matrix(), pauliz_matrix())
    sum = empty()
    assert np.allclose(sum.to_matrix(), zero_matrix(1))
    sum -= identity(0)
    assert sum.degrees() == [0]
    assert np.allclose(sum.to_matrix(), -identity_matrix(2))
    ids = identities(3, 5)
    assert ids.degrees() == [3, 4]
    assert np.allclose(ids.to_matrix(), identity_matrix(2 * 2))
    canon = ids.copy().canonicalize()
    assert ids.degrees() == [3, 4]
    assert canon.degrees() == []
    assert canon.to_matrix() == identity_matrix(1)

    # construction from Pauli word
    want_spin_op = z(0) * z(1) * z(2)
    got_spin_op = SpinOperator.from_word("ZZZ")
    assert got_spin_op == want_spin_op
    want_spin_op = x(0) * x(1) * i(2) * x(3) * y(4) * z(5)
    got_spin_op = SpinOperator.from_word("XXIXYZ")
    assert got_spin_op == want_spin_op
    want_spin_op = x(0) * y(1) * z(2)
    got_spin_op = SpinOperator.from_word("XYZ")
    assert got_spin_op == want_spin_op

    # construction of random operator
    qubit_count = 5
    term_count = 7
    for seed in range(100):
        hamiltonian = SpinOperator.random(qubit_count, term_count, seed=seed)
        assert hamiltonian.get_term_count() == term_count
    qubit_count = 3
    term_count = 21  # too many because 6 choose 3 = 20
    with pytest.raises(RuntimeError) as error:
        hamiltonian = SpinOperator.random(qubit_count, term_count)


def test_iteration():
    prod1 = x(1) * y(0)
    prod2 = y(0) * z(1)
    sum = prod1 + prod2
    for p1, p2 in zip(sum, [prod1, prod2]):
        for t1, t2 in zip(p1, p2):
            assert t1 == t2
    sum_terms = 0
    prod_terms = 0
    for prod in sum:
        sum_terms += 1
        term_id = ""
        for op in prod:
            prod_terms += 1
            term_id += op.to_string(include_degrees = True)
        assert term_id == prod.get_term_id()
    assert sum_terms == 2
    assert prod_terms == 4

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    count = 0
    for term in hamiltonian:
        count += 1
    assert count == 5


def test_properties():

    prod1 = (1. + 0.5j) * x(1) * y(0)
    prod2 = z(1) * x(3)
    sum = prod1 + prod2
    assert prod1.degrees() == [0, 1]
    assert prod2.degrees() == [1, 3]
    assert sum.degrees() == [0, 1, 3]
    assert prod1.get_min_degree() == 0
    assert prod1.get_max_degree() == 1
    assert prod2.get_min_degree() == 1
    assert prod2.get_max_degree() == 3
    assert sum.get_min_degree() == 0
    assert sum.get_max_degree() == 3

    assert sum.get_term_count() == 2
    assert prod1.get_ops_count() == 2
    sum += prod1
    assert sum.get_term_count() == 2
    prod1_mat = (1. + 0.5j) * np.kron(identity_matrix(2), np.kron(paulix_matrix(), pauliy_matrix()))
    prod2_mat = np.kron(paulix_matrix(), np.kron(pauliz_matrix(), identity_matrix(2)))
    assert np.allclose(sum.to_matrix(), prod1_mat + prod1_mat + prod2_mat)
    assert str(prod1) == "(1.000000+0.500000i) * Y0X1"
    assert str(sum) == "(2.000000+1.000000i) * Y0X1 + (1.000000+0.000000i) * Z1X3"
    assert prod1.get_term_id() == "Y0X1"

    spin_operator = empty()
    # (is_identity on sum is deprecated, kept for backwards compatibility)
    assert spin_operator.is_identity()
    # Sum is empty.
    assert spin_operator.get_term_count() == 0
    assert spin_operator.get_qubit_count() == 0
    spin_operator -= x(1)
    # Should now have 1 term acting on 1 qubit.
    assert spin_operator.get_term_count() == 1
    assert spin_operator.get_qubit_count() == 1
    # No longer identity.
    assert not spin_operator.is_identity()
    # Term should have a coefficient -1
    term, *_ = spin_operator
    assert term.evaluate_coefficient() == -1.0
    assert term.get_coefficient() == -1.0 # deprecated function replaced by evaluate_coefficient


def test_matrix_construction():
    # dense matrix
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    mat = hamiltonian.to_matrix()
    assert abs(-1.74 - np.min(np.linalg.eigvals(mat))) < 1e-2
    print(mat)
    want_matrix = np.array([[.00029, 0, 0, 0], [0, -.43629, -4.2866, 0],
                            [0, -4.2866, 12.2503, 0], [0, 0, 0, 11.8137]])
    got_matrix = np.array(mat, copy=False)
    assert np.allclose(want_matrix, got_matrix, rtol=1e-3)

    # sparse matrix
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


def test_canonicalization():
    all_degrees = [0, 1, 2, 3]

    # product operator
    for id_target in all_degrees:
        op = identity()
        expected = identity()
        for target in all_degrees:
            if target == id_target:
                op *= identity(target)
            elif target % 2 == 0:
                op *= x(target)
                expected *= x(target)
            else:
                op *= z(target)
                expected *= z(target)

        assert op != expected
        assert op.degrees() == all_degrees
        op.canonicalize()
        assert op == expected
        assert op.degrees() != all_degrees
        assert op.degrees() == expected.degrees()
        assert np.allclose(op.to_matrix(), expected.to_matrix())

        op.canonicalize(set(all_degrees))
        assert op.degrees() == all_degrees
        canon = canonicalized(op)
        assert op.degrees() == all_degrees
        assert canon.degrees() == expected.degrees()

    # sum operator
    previous = empty()
    expected = empty()
    def check_expansion(got, want_degrees):
        canon = got.copy() # standard python behavior is for assignments not to copy
        term_with_missing_degrees = False
        for term in canon:
            if term.degrees() != all_degrees:
                term_with_missing_degrees = True
        assert term_with_missing_degrees
        assert canon == got
        canon.canonicalize(want_degrees)
        assert canon != got
        assert canon.degrees() == all_degrees
        for term in canon:
            assert term.degrees() == all_degrees

    for id_target in all_degrees:
        term = identity()
        expected_term = identity()
        for target in all_degrees:
            if target == id_target:
                term *= identity(target)
            elif target & 2:
                term *= z(target)
                expected_term *= z(target)
            else:
                term *= x(target)
                expected_term *= x(target)
        previous += term
        expected += expected_term
        got = previous

        assert got != expected
        assert canonicalized(got) == expected
        assert got != expected
        got.canonicalize()
        assert got == expected
        assert got.degrees() == expected.degrees()
        assert np.allclose(got.to_matrix(), expected.to_matrix())
        check_expansion(got, set(all_degrees))
        if id_target > 0: check_expansion(got, set())
        with pytest.raises(Exception):
            got.canonicalize(got.degrees()[1:])


def test_trimming():
    all_degrees = [idx for idx in range(6)]
    for _ in range(10):
        bit_mask = random.getrandbits(len(all_degrees))
        expected = empty()
        terms = [identity()] * len(all_degrees)
        # randomize order in which we add terms
        term_order = np.random.permutation(range(len(all_degrees)))
        for idx in range(len(all_degrees)):
            coeff = (bit_mask >> idx) & 1
            prod = identity(all_degrees[idx]) * float(coeff)
            if coeff > 0:
                expected += prod
            terms[term_order[idx]] = prod
        orig = empty()
        for term in terms:
            orig += term
        assert orig.get_term_count() == len(all_degrees)
        assert orig.degrees() == all_degrees
        orig.trim()
        assert orig.get_term_count() < len(all_degrees)
        assert orig.get_term_count() == expected.get_term_count()
        assert orig.degrees() == expected.degrees()
        assert np.allclose(orig.to_matrix(), expected.to_matrix())
        # check that our term map seems accurate
        for term in expected:
            orig += float(term.degrees()[0]) * term
        assert orig.get_term_count() == expected.get_term_count()
        assert orig.degrees() == expected.degrees()
        for term in orig:
            assert term.evaluate_coefficient() == term.degrees()[0] + 1.


def test_equality():
    prod1 = x(0) * y(0)
    prod2 = x(1) * y(1)
    prod3 = x(0) * y(1)
    prod4 = y(1) * x(0)
    sum = SpinOperator(prod1)
    assert prod1 != prod2
    assert prod3 == prod4
    assert sum == prod1
    sum += prod3
    assert sum != prod1
    assert sum == (prod3 + prod1)
    sum += prod1
    assert sum != (prod3 + prod1)
    assert sum == (prod3 + 2. * prod1)
    assert sum != sum + 1.
    assert sum != i(2) * sum
    assert np.allclose(np.kron(identity_matrix(2), sum.to_matrix()), (i(2) * sum).to_matrix())

    op1 = 5.907 - 2.1433 * spin.x(0) * spin.x(1) + spin.y(0) * spin.y(1)
    op2 = 3.1433 * spin.y(0) * spin.y(1) - .21829 * spin.z(0) + 6.125 * spin.z(
        1)
    op = op1 - op2
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    assert hamiltonian == op


def test_arithmetics():
    # basic tests for all arithmetic related bindings - 
    # more complex expressions are tested as part of the C++ tests
    dims = {0: 2, 1: 2}
    id = i(0)
    sum = y(0) + z(1)
    sum_matrix = np.kron(pauliz_matrix(), identity_matrix(2)) +\
                 np.kron(identity_matrix(2), pauliy_matrix())
    assert np.allclose(id.to_matrix(dims), identity_matrix(2))
    assert np.allclose(sum.to_matrix(dims), sum_matrix)

    # unary operators
    assert np.allclose((-id).to_matrix(dims), -1. * identity_matrix(2))
    assert np.allclose((-sum).to_matrix(dims), -1. * sum_matrix)
    assert np.allclose(id.to_matrix(dims), identity_matrix(2))
    assert np.allclose(sum.to_matrix(dims), sum_matrix)
    assert np.allclose((+id).canonicalize().to_matrix(dims), identity_matrix(1))
    assert np.allclose((+sum).canonicalize().to_matrix({0: 2}), sum_matrix)
    assert np.allclose(id.to_matrix({0: 2}), identity_matrix(2))

    # right-hand arithmetics
    assert np.allclose((id * 2.).to_matrix(), 2. * identity_matrix(2))
    assert np.allclose((sum * 2.).to_matrix(), 2. * sum_matrix)
    assert np.allclose((id * 2.j).to_matrix(), 2.j * identity_matrix(2))
    assert np.allclose((sum * 2.j).to_matrix(), 2.j * sum_matrix)
    assert np.allclose((sum * id).to_matrix(), sum_matrix)
    assert np.allclose((id * sum).to_matrix(), sum_matrix)
    assert np.allclose((id + 2.).to_matrix(), 3. * identity_matrix(2))
    assert np.allclose((sum + 2.).to_matrix(), sum_matrix + 2. * identity_matrix(2 * 2))
    assert np.allclose((id + 2.j).to_matrix(), (1. + 2.j) * identity_matrix(2))
    assert np.allclose((sum + 2.j).to_matrix(), sum_matrix + 2.j * identity_matrix(2 * 2))
    assert np.allclose((sum + id).to_matrix(), sum_matrix + identity_matrix(2 * 2))
    assert np.allclose((id + sum).to_matrix(), sum_matrix + identity_matrix(2 * 2))
    assert np.allclose((id - 2.).to_matrix(), -1. * identity_matrix(2))
    assert np.allclose((sum - 2.).to_matrix(), sum_matrix - 2. * identity_matrix(2 * 2))
    assert np.allclose((id - 2.j).to_matrix(), (1. - 2.j) * identity_matrix(2))
    assert np.allclose((sum - 2.j).to_matrix(), sum_matrix - 2.j * identity_matrix(2 * 2))
    assert np.allclose((sum - id).to_matrix(), sum_matrix - identity_matrix(2 * 2))
    assert np.allclose((id - sum).to_matrix(), identity_matrix(2 * 2) - sum_matrix)

    # in-place arithmetics
    term = id.copy()
    op = +sum
    term *= 2.
    op *= 2.
    assert np.allclose(term.to_matrix(), 2. * identity_matrix(2))
    assert np.allclose(op.to_matrix(), 2. * sum_matrix)
    term *= 0.5j
    op *= 0.5j
    assert np.allclose(term.to_matrix(), 1.j * identity_matrix(2))
    assert np.allclose(op.to_matrix(), 1.j * sum_matrix)
    op *= term
    assert np.allclose(op.to_matrix(), -1. * sum_matrix)

    op += 2.
    assert np.allclose(op.to_matrix(), -1. * sum_matrix + 2. * identity_matrix(2 * 2))
    op += term
    assert np.allclose(op.to_matrix(), -1. * sum_matrix + (2. + 1.j) * identity_matrix(2 * 2))
    op -= 2.
    assert np.allclose(op.to_matrix(), -1. * sum_matrix + 1.j * identity_matrix(2 * 2))
    op -= term
    assert np.allclose(op.to_matrix(), -1. * sum_matrix)

    # left-hand arithmetics
    assert np.allclose((2. * id).to_matrix(), 2. * identity_matrix(2))
    assert np.allclose((2. * sum).to_matrix(), 2. * sum_matrix)
    assert np.allclose((2.j * id).to_matrix(), 2.j * identity_matrix(2))
    assert np.allclose((2.j * sum).to_matrix(), 2.j * sum_matrix)
    assert np.allclose((2. + id).to_matrix(), 3. * identity_matrix(2))
    assert np.allclose((2. + sum).to_matrix(), sum_matrix + 2. * identity_matrix(2 * 2))
    assert np.allclose((2.j + id).to_matrix(), (1 + 2j) * identity_matrix(2))
    assert np.allclose((2.j + sum).to_matrix(), sum_matrix + 2.j * identity_matrix(2 * 2))
    assert np.allclose((2. - id).to_matrix(), identity_matrix(2))
    assert np.allclose((2. - sum).to_matrix(), 2. * identity_matrix(2 * 2) - sum_matrix)
    assert np.allclose((2.j - id).to_matrix(), (-1 + 2.j) * identity_matrix(2))
    assert np.allclose((2.j - sum).to_matrix(), 2.j * identity_matrix(2 * 2) - sum_matrix)

    # legacy tests

    # Test the empty constructor.
    spin_a = empty()
    spin_b = x(0)
    # Test the copy constructor.
    spin_b_copy = SpinOperator(spin_b)
    assert (spin_b_copy == spin_b)
    assert (spin_b_copy != spin_a)
    spin_c = y(1)
    spin_d = z(2)

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
    # this was 3 due to the (incorrect) identity that the default constructor used to create
    # same goes for all other len check adjustments in this test
    assert (len(data) == 2)
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
    assert (len(data) == 3)
    assert (len(data[0]) == 6)
    expected = [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 0]]
    assert (all([d in expected for d in data]))
    expected = [5 + 5j, 5 + 5j, -5 - 5j, 1]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_g.get_raw_data()
    assert (len(data) == 3)
    assert (len(data[0]) == 6)
    expected = [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 0]]
    assert (all([d in expected for d in data]))
    expected = [5 + 5j, 5 + 5j, -5 - 5j, -1]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_h.get_raw_data()
    assert (len(data) == 2)
    assert (len(data[0]) == 6)
    expected = [[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [5 + 5j, 5 + 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_i.get_raw_data()
    assert (len(data) == 2)
    assert (len(data[0]) == 6)
    expected = [[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [-5 - 5j, 5 + 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_j.get_raw_data()
    assert (len(data) == 2)
    assert (len(data[0]) == 6)
    expected = [[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [-5 - 5j, 5 + 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_k.get_raw_data()
    assert (len(data) == 2)
    assert (len(data[0]) == 6)
    expected = [[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [10j, 10j, -10j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_l.get_raw_data()
    assert (len(data) == 2)
    assert (len(data[0]) == 6)
    expected = [[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [10j, 10j, -10j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_m.get_raw_data()
    assert (len(data) == 3)
    assert (len(data[0]) == 6)
    expected = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [3, 5 + 5j, 5 + 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_n.get_raw_data()
    assert (len(data) == 3)
    assert (len(data[0]) == 6)
    expected = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [3, 5 + 5j, 5 + 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_o.get_raw_data()
    assert (len(data) == 3)
    assert (len(data[0]) == 6)
    expected = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [-3, 5 + 5j, 5 + 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))

    data, coeffs = spin_p.get_raw_data()
    assert (len(data) == 3)
    assert (len(data[0]) == 6)
    expected = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 1]]
    assert (all([d in expected for d in data]))
    expected = [3, 5 + 5j, -5 - 5j, -5 - 5j]
    assert (all([c in expected for c in coeffs]))


def test_term_distribution():
    op = empty()
    for target in range(7):
        op += i(target)
    batches = op.distribute_terms(4)
    assert op.get_term_count() == 7
    assert len(batches) == 4
    for idx in range(2):
        assert batches[idx].get_term_count() == 2
    assert batches[3].get_term_count() == 1
    sum = empty()
    for batch in batches:
        sum += batch
    assert sum == op

    qubit_count = 7
    term_count = 10
    num_of_gpus = 3
    hamiltonian = SpinOperator.random(qubit_count, term_count, seed=13)
    batched = hamiltonian.distribute_terms(num_of_gpus)
    assert len(batched) == 3
    assert batched[0].get_term_count() == 4
    assert batched[1].get_term_count() == 3
    assert batched[2].get_term_count() == 3


def test_serialization():
    for nq in range(1, 31):
        for nt in range(1, nq + 1):
            # random will product terms that each act on all
            # qubits in the range [0, nq)
            h1 = SpinOperator.random(qubit_count=nq,
                                           term_count=nt,
                                           seed=13)
            h2 = h1.serialize()
            h3 = SpinOperator(h2)
            assert (h1 == h3)


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


# deprecated functionality - replaced by iteration
def test_legacy_foreach():
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
        def doSomethingWithPauli(pauli: Pauli, idx: int):
            nonlocal counter, xSupports
            if pauli == Pauli.X:
                counter = counter + 1
                xSupports.append(idx)
        term.for_each_pauli(doSomethingWithPauli)

    hamiltonian.for_each_term(doSomethingWithTerm)
    assert counter == 2
    assert xSupports == [0, 1]


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
