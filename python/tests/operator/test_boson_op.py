# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np, pytest, random
from cudaq import boson
from cudaq.operators.boson import *
from op_utils import *  # test helpers

has_scipy = True
try:
    import scipy
except ImportError:
    has_scipy = False


@pytest.fixture(autouse=True)
def setup():
    random.seed(10)
    yield


def test_definitions():
    dims = {0: 2, 1: 3}

    assert np.allclose(boson.create(1).to_matrix(dims), create_matrix(3))
    assert np.allclose(
        boson.annihilate(1).to_matrix(dims), annihilate_matrix(3))
    assert np.allclose(boson.number(1).to_matrix(dims), number_matrix(3))
    assert np.allclose(boson.position(1).to_matrix(dims), position_matrix(3))
    assert np.allclose(boson.momentum(1).to_matrix(dims), momentum_matrix(3))

    assert np.allclose(create(1).to_matrix(dims), create_matrix(3))
    assert np.allclose(annihilate(1).to_matrix(dims), annihilate_matrix(3))
    assert np.allclose(number(1).to_matrix(dims), number_matrix(3))
    assert np.allclose(position(1).to_matrix(dims), position_matrix(3))
    assert np.allclose(momentum(1).to_matrix(dims), momentum_matrix(3))


def test_commutation_relations():

    dims = {0: 4, 1: 4}
    ad_mat = create_matrix(5)
    a_mat = annihilate_matrix(5)
    padded_commutator = np.dot(a_mat, ad_mat) - np.dot(ad_mat, a_mat)
    commutator_mat = padded_commutator[:4, :4]
    assert np.allclose(commutator_mat, identity_matrix(4))

    # Expected commutation relations:
    # [a(k), a†(q)] = δkq
    # [a†(k), a†(q)] = [a(k), a(q)] = 0
    def commutator(ad, a):
        return a * ad - ad * a

    # check [a(q), a†(q)] = 1
    rel1 = commutator(create(0), annihilate(0))
    rel2 = commutator(create(1), annihilate(1))
    assert np.allclose(rel1.to_matrix(dims), identity_matrix(4))
    assert np.allclose(rel2.to_matrix(dims), identity_matrix(4))

    # check [a(k), a†(q)] = 0 for k != q
    rel1 = commutator(create(0), annihilate(1))
    rel2 = commutator(create(1), annihilate(0))
    assert np.allclose(rel1.to_matrix(dims), zero_matrix(16))
    assert np.allclose(rel2.to_matrix(dims), zero_matrix(16))

    # check [a†(q), a†(q)] = 0
    rel1 = commutator(create(0), create(0))
    rel2 = commutator(create(1), create(1))
    assert np.allclose(rel1.to_matrix(dims), zero_matrix(4))
    assert np.allclose(rel2.to_matrix(dims), zero_matrix(4))

    # check [a(q), a(q)] = 0
    rel1 = commutator(annihilate(0), annihilate(0))
    rel2 = commutator(annihilate(1), annihilate(1))
    assert np.allclose(rel1.to_matrix(dims), zero_matrix(4))
    assert np.allclose(rel2.to_matrix(dims), zero_matrix(4))

    # check [a†(k), a†(q)] = 0 for k != q
    rel1 = commutator(create(0), create(1))
    rel2 = commutator(create(1), create(0))
    assert np.allclose(rel1.to_matrix(dims), zero_matrix(16))
    assert np.allclose(rel2.to_matrix(dims), zero_matrix(16))

    # check [a(k), a(q)] = 0 for k != q
    rel1 = commutator(annihilate(0), annihilate(1))
    rel2 = commutator(annihilate(1), annihilate(0))
    assert np.allclose(rel1.to_matrix(dims), zero_matrix(16))
    assert np.allclose(rel2.to_matrix(dims), zero_matrix(16))


def test_construction():
    prod = identity()
    assert np.allclose(prod.to_matrix(), identity_matrix(1))
    prod *= number(0)
    assert np.allclose(prod.to_matrix({0: 3}), number_matrix(3))
    sum = empty()
    assert np.allclose(sum.to_matrix(), zero_matrix(1))
    sum *= number(0)
    assert sum.degrees == []
    assert np.allclose(sum.to_matrix(), zero_matrix(1))
    sum += identity(1)
    assert sum.degrees == [1]
    assert np.allclose(sum.to_matrix({1: 3}), identity_matrix(3))
    sum *= number(1)
    assert np.allclose(sum.to_matrix({1: 3}), number_matrix(3))
    sum = empty()
    assert np.allclose(sum.to_matrix(), zero_matrix(1))
    sum -= identity(0)
    assert sum.degrees == [0]
    assert np.allclose(sum.to_matrix({0: 3}), -identity_matrix(3))
    ids = identities(3, 5)
    assert ids.degrees == [3, 4]
    assert np.allclose(ids.to_matrix({3: 3, 4: 3}), identity_matrix(3 * 3))
    canon = ids.copy().canonicalize()
    assert ids.degrees == [3, 4]
    assert canon.degrees == []
    assert canon.to_matrix() == identity_matrix(1)


def test_iteration():
    prod1 = annihilate(1) * create(0)
    prod2 = number(0) * create(1)
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
            term_id += op.to_string(include_degrees=True)
        assert term_id == prod.term_id
    assert sum_terms == 2
    assert prod_terms == 4


def test_properties():

    prod1 = position(1) * momentum(0)
    prod2 = number(1) * annihilate(3)
    sum = prod1 + prod2
    assert prod1.degrees == [0, 1]
    assert prod2.degrees == [1, 3]
    assert sum.degrees == [0, 1, 3]
    assert prod1.min_degree == 0
    assert prod1.max_degree == 1
    assert prod2.min_degree == 1
    assert prod2.max_degree == 3
    assert sum.min_degree == 0
    assert sum.max_degree == 3

    dims = {0: 2, 1: 3, 2: 2, 3: 4}
    assert sum.term_count == 5  # position and momentum are a sum of two terms each
    assert prod2.ops_count == 2
    sum += prod1
    assert sum.term_count == 5
    prod1_mat = np.kron(identity_matrix(4),
                        np.kron(position_matrix(3), momentum_matrix(2)))
    prod2_mat = np.kron(annihilate_matrix(4),
                        np.kron(number_matrix(3), identity_matrix(2)))
    assert np.allclose(sum.to_matrix(dims), prod1_mat + prod1_mat + prod2_mat)

    prod1.dump()
    sum.dump()
    assert str(
        prod1
    ) == "(0+0.25i) * Ad0Ad1 + (0-0.25i) * A0Ad1 + (0+0.25i) * Ad0A1 + (0-0.25i) * A0A1"
    assert str(
        sum
    ) == "(0+0.5i) * Ad0Ad1 + (0-0.5i) * A0Ad1 + (0+0.5i) * Ad0A1 + (0-0.5i) * A0A1 + (1+0i) * N1A3"
    assert prod2.term_id == "N1A3"


def test_matrix_construction():
    dims = {0: 3, 1: 4}
    hamiltonian = 5.5 - 2.03 * create(0) * annihilate(1) - 2.33 * number(0)
    mat = hamiltonian.to_matrix(dims)
    ev = sorted(np.linalg.eigvals(mat))

    # sparse matrix
    data, rows, cols = hamiltonian.to_sparse_matrix(dims)
    for i, value in enumerate(data):
        print(rows[i], cols[i], value)
        assert np.isclose(mat[rows[i], cols[i]], value)
    if has_scipy:
        scipyM = scipy.sparse.csr_array((data, (rows, cols)),
                                        shape=(3 * 4, 3 * 4))
        scipyEv = scipy.sparse.linalg.eigs(scipyM,
                                           k=10,
                                           return_eigenvectors=False,
                                           sigma=ev[0] - 1e-2)
        assert np.allclose(ev[:10], sorted(scipyEv), rtol=1e-2)


def test_canonicalization():
    dims = {0: 2, 1: 3, 2: 2, 3: 4}
    all_degrees = [0, 1, 2, 3]

    # product operator
    for id_target in all_degrees:
        op = identity()
        expected = identity()
        for target in all_degrees:
            if target == id_target:
                op *= identity(target)
            elif target % 2 == 0:
                op *= create(target)
                expected *= create(target)
            else:
                op *= annihilate(target)
                expected *= annihilate(target)

        assert op != expected
        assert op.degrees == all_degrees
        op.canonicalize()
        assert op == expected
        assert op.degrees != all_degrees
        assert op.degrees == expected.degrees
        assert np.allclose(op.to_matrix(dims), expected.to_matrix(dims))

        op.canonicalize(set(all_degrees))
        assert op.degrees == all_degrees
        canon = canonicalized(op)
        assert op.degrees == all_degrees
        assert canon.degrees == expected.degrees

    # sum operator
    previous = empty()
    expected = empty()

    def check_expansion(got, want_degrees):
        canon = got.copy(
        )  # standard python behavior is for assignments not to copy
        term_with_missing_degrees = False
        for term in canon:
            if term.degrees != all_degrees:
                term_with_missing_degrees = True
        assert term_with_missing_degrees
        assert canon == got
        canon.canonicalize(want_degrees)
        assert canon != got
        assert canon.degrees == all_degrees
        for term in canon:
            assert term.degrees == all_degrees

    for id_target in all_degrees:
        term = BosonOperator()
        expected_term = BosonOperator()
        for target in all_degrees:
            if target == id_target:
                term *= identity(target)
            elif target & 2:
                term *= position(target)
                expected_term *= position(target)
            else:
                term *= momentum(target)
                expected_term *= momentum(target)
        previous += term
        expected += expected_term
        got = previous

        assert got != expected
        assert canonicalized(got) == expected
        assert got != expected
        got.canonicalize()
        assert got == expected
        assert got.degrees == expected.degrees
        assert np.allclose(got.to_matrix(dims), expected.to_matrix(dims))
        check_expansion(got, set(all_degrees))
        if id_target > 0:
            check_expansion(got, set())
        with pytest.raises(Exception):
            got.canonicalize(got.degrees[1:])


def test_trimming():
    all_degrees = [idx for idx in range(6)]
    dims = dict(((d, 2) for d in all_degrees))
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
        assert orig.term_count == len(all_degrees)
        assert orig.degrees == all_degrees
        orig.trim()
        assert orig.term_count < len(all_degrees)
        assert orig.term_count == expected.term_count
        assert orig.degrees == expected.degrees
        assert np.allclose(orig.to_matrix(dims), expected.to_matrix(dims))
        # check that our term map seems accurate
        for term in expected:
            orig += float(term.degrees[0]) * term
        assert orig.term_count == expected.term_count
        assert orig.degrees == expected.degrees
        for term in orig:
            assert term.evaluate_coefficient() == term.degrees[0] + 1.


def test_equality():
    prod1 = position(0) * momentum(0)
    prod2 = position(1) * momentum(1)
    prod3 = position(0) * momentum(1)
    prod4 = momentum(1) * position(0)
    sum = BosonOperator(prod1)
    assert prod1 != prod2
    assert prod3 == prod4
    assert sum == prod1
    assert prod1 == sum
    sum += prod3
    assert sum != prod1
    assert sum == (prod3 + prod1)
    sum += prod1
    assert sum != (prod3 + prod1)
    assert sum == (prod3 + 2. * prod1)
    assert sum != sum + 1.
    assert sum != identity(2) * sum
    dims = {0: 2, 1: 3, 2: 2, 3: 4}
    assert np.allclose(np.kron(identity_matrix(2), sum.to_matrix(dims)),
                       (identity(2) * sum).to_matrix(dims))


def test_arithmetics():
    # basic tests for all arithmetic related bindings -
    # more complex expressions are tested as part of the C++ tests
    dims = {0: 3, 1: 2}
    id = identity(0)
    sum = momentum(0) + position(1)
    sum_matrix = np.kron(position_matrix(2), identity_matrix(3)) +\
                 np.kron(identity_matrix(2), momentum_matrix(3))
    assert np.allclose(id.to_matrix(dims), identity_matrix(3))
    assert np.allclose(sum.to_matrix(dims), sum_matrix)

    # unary operators
    assert np.allclose((-id).to_matrix(dims), -1. * identity_matrix(3))
    assert np.allclose((-sum).to_matrix(dims), -1. * sum_matrix)
    assert np.allclose(id.to_matrix(dims), identity_matrix(3))
    assert np.allclose(sum.to_matrix(dims), sum_matrix)
    assert np.allclose((+id).canonicalize().to_matrix(), identity_matrix(1))
    assert np.allclose((+sum).canonicalize().to_matrix(dims), sum_matrix)
    assert np.allclose(id.to_matrix(dims), identity_matrix(3))

    # right-hand arithmetics
    assert np.allclose((id * 2.).to_matrix(dims), 2. * identity_matrix(3))
    assert np.allclose((sum * 2.).to_matrix(dims), 2. * sum_matrix)
    assert np.allclose((id * 2.j).to_matrix(dims), 2.j * identity_matrix(3))
    assert np.allclose((sum * 2.j).to_matrix(dims), 2.j * sum_matrix)
    assert np.allclose((sum * id).to_matrix(dims), sum_matrix)
    assert np.allclose((id * sum).to_matrix(dims), sum_matrix)
    assert np.allclose((id + 2.).to_matrix(dims), 3. * identity_matrix(3))
    assert np.allclose((sum + 2.).to_matrix(dims),
                       sum_matrix + 2. * identity_matrix(2 * 3))
    assert np.allclose((id + 2.j).to_matrix(dims),
                       (1. + 2.j) * identity_matrix(3))
    assert np.allclose((sum + 2.j).to_matrix(dims),
                       sum_matrix + 2.j * identity_matrix(2 * 3))
    assert np.allclose((sum + id).to_matrix(dims),
                       sum_matrix + identity_matrix(2 * 3))
    assert np.allclose((id + sum).to_matrix(dims),
                       sum_matrix + identity_matrix(2 * 3))
    assert np.allclose((id - 2.).to_matrix(dims), -1. * identity_matrix(3))
    assert np.allclose((sum - 2.).to_matrix(dims),
                       sum_matrix - 2. * identity_matrix(2 * 3))
    assert np.allclose((id - 2.j).to_matrix(dims),
                       (1. - 2.j) * identity_matrix(3))
    assert np.allclose((sum - 2.j).to_matrix(dims),
                       sum_matrix - 2.j * identity_matrix(2 * 3))
    assert np.allclose((sum - id).to_matrix(dims),
                       sum_matrix - identity_matrix(2 * 3))
    assert np.allclose((id - sum).to_matrix(dims),
                       identity_matrix(2 * 3) - sum_matrix)

    # in-place arithmetics
    term = id.copy()
    op = +sum
    term *= 2.
    op *= 2.
    assert np.allclose(term.to_matrix(dims), 2. * identity_matrix(3))
    assert np.allclose(op.to_matrix(dims), 2. * sum_matrix)
    term *= 0.5j
    op *= 0.5j
    assert np.allclose(term.to_matrix(dims), 1.j * identity_matrix(3))
    assert np.allclose(op.to_matrix(dims), 1.j * sum_matrix)
    op *= term
    assert np.allclose(op.to_matrix(dims), -1. * sum_matrix)

    op += 2.
    assert np.allclose(op.to_matrix(dims),
                       -1. * sum_matrix + 2. * identity_matrix(2 * 3))
    op += term
    assert np.allclose(op.to_matrix(dims),
                       -1. * sum_matrix + (2. + 1.j) * identity_matrix(2 * 3))
    op -= 2.
    assert np.allclose(op.to_matrix(dims),
                       -1. * sum_matrix + 1.j * identity_matrix(2 * 3))
    op -= term
    assert np.allclose(op.to_matrix(dims), -1. * sum_matrix)

    # left-hand arithmetics
    assert np.allclose((2. * id).to_matrix(dims), 2. * identity_matrix(3))
    assert np.allclose((2. * sum).to_matrix(dims), 2. * sum_matrix)
    assert np.allclose((2.j * id).to_matrix(dims), 2.j * identity_matrix(3))
    assert np.allclose((2.j * sum).to_matrix(dims), 2.j * sum_matrix)
    assert np.allclose((2. + id).to_matrix(dims), 3. * identity_matrix(3))
    assert np.allclose((2. + sum).to_matrix(dims),
                       sum_matrix + 2. * identity_matrix(2 * 3))
    assert np.allclose((2.j + id).to_matrix(dims),
                       (1 + 2j) * identity_matrix(3))
    assert np.allclose((2.j + sum).to_matrix(dims),
                       sum_matrix + 2.j * identity_matrix(2 * 3))
    assert np.allclose((2. - id).to_matrix(dims), identity_matrix(3))
    assert np.allclose((2. - sum).to_matrix(dims),
                       2. * identity_matrix(2 * 3) - sum_matrix)
    assert np.allclose((2.j - id).to_matrix(dims),
                       (-1 + 2.j) * identity_matrix(3))
    assert np.allclose((2.j - sum).to_matrix(dims),
                       2.j * identity_matrix(2 * 3) - sum_matrix)


def test_term_distribution():
    op = empty()
    for target in range(7):
        op += identity(target)
    batches = op.distribute_terms(4)
    assert op.term_count == 7
    assert len(batches) == 4
    for idx in range(3):
        assert batches[idx].term_count == 2
    assert batches[3].term_count == 1
    sum = empty()
    for batch in batches:
        sum += batch
    assert sum == op


# Run with: pytest -rP
if __name__ == "__main__":
    pytest.main(["-rP"])
