# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cmath, numpy as np, pytest, random
from cudaq import operators
from cudaq.operators import *
from op_utils import *  # test helpers


@pytest.fixture(autouse=True)
def setup():
    random.seed(10)
    yield


def test_definitions():
    dims = {0: 2, 1: 3}
    assert np.allclose(operators.number(1).to_matrix(dims), number_matrix(3))
    assert np.allclose(operators.parity(1).to_matrix(dims), parity_matrix(3))
    assert np.allclose(
        operators.position(1).to_matrix(dims), position_matrix(3))
    assert np.allclose(
        operators.momentum(1).to_matrix(dims), momentum_matrix(3))
    assert np.allclose(
        operators.squeeze(1).to_matrix(dims,
                                       squeezing=0.5 + 1.2j,
                                       displacement=0.5 + 1.2j),
        squeeze_matrix(3, 0.5 + 1.2j))
    assert np.allclose(
        operators.displace(1).to_matrix(dims,
                                        squeezing=0.5 + 1.2j,
                                        displacement=0.5 + 1.2j),
        displace_matrix(3, 0.5 + 1.2j))
    params = {"squeezing": 0.5 + 1.2j, "displacement": 0.5 + 1.2j}
    assert np.allclose(
        operators.squeeze(1).to_matrix(dims, params),
        squeeze_matrix(3, 0.5 + 1.2j))
    assert np.allclose(
        operators.displace(1).to_matrix(dims, params),
        displace_matrix(3, 0.5 + 1.2j))
    with pytest.raises(Exception):
        operators.squeeze(1).to_matrix(dims, displacement=0.5 + 1.2j)
    with pytest.raises(Exception):
        operators.displace(1).to_matrix(dims, squeeze=0.5 + 1.2j)

    assert np.allclose(number(1).to_matrix(dims), number_matrix(3))
    assert np.allclose(parity(1).to_matrix(dims), parity_matrix(3))
    assert np.allclose(position(1).to_matrix(dims), position_matrix(3))
    assert np.allclose(momentum(1).to_matrix(dims), momentum_matrix(3))
    assert np.allclose(
        squeeze(1).to_matrix(dims,
                             squeezing=0.5 + 1.2j,
                             displacement=0.5 + 1.2j),
        squeeze_matrix(3, 0.5 + 1.2j))
    assert np.allclose(
        displace(1).to_matrix(dims,
                              squeezing=0.5 + 1.2j,
                              displacement=0.5 + 1.2j),
        displace_matrix(3, 0.5 + 1.2j))
    params = {"squeezing": 0.5 + 1.2j, "displacement": 0.5 + 1.2j}
    assert np.allclose(
        squeeze(1).to_matrix(dims, params), squeeze_matrix(3, 0.5 + 1.2j))
    assert np.allclose(
        displace(1).to_matrix(dims, params), displace_matrix(3, 0.5 + 1.2j))
    with pytest.raises(Exception):
        squeeze(1).to_matrix(dims, displacement=0.5 + 1.2j)
    with pytest.raises(Exception):
        displace(1).to_matrix(dims, squeeze=0.5 + 1.2j)

    squeeze_params = squeeze(1).parameters
    print(squeeze_params)
    assert "squeezing" in squeeze_params
    assert squeeze_params["squeezing"] != ""

    displace_params = displace(1).parameters
    print(displace_params)
    assert "displacement" in displace_params
    assert displace_params["displacement"] != ""


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
    prod1 = position(1) * momentum(0)
    prod2 = number(0) * parity(0)
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
    prod2 = number(1) * parity(3)
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
    assert sum.term_count == 2
    assert prod1.ops_count == 2
    sum += prod1
    assert sum.term_count == 2
    prod1_mat = np.kron(identity_matrix(4),
                        np.kron(position_matrix(3), momentum_matrix(2)))
    prod2_mat = np.kron(parity_matrix(4),
                        np.kron(number_matrix(3), identity_matrix(2)))
    assert np.allclose(sum.to_matrix(dims), prod1_mat + prod1_mat + prod2_mat)

    prod1.dump()
    sum.dump()
    assert str(prod1) == "(1+0i) * momentum(0)position(1)"
    assert str(
        sum) == "(2+0i) * momentum(0)position(1) + (1+0i) * number(1)parity(3)"
    assert prod1.term_id == "momentum(0)position(1)"


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
                op *= parity(target)
                expected *= parity(target)
            else:
                op *= number(target)
                expected *= number(target)

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
        term = identity()
        expected_term = identity()
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
    sum = MatrixOperator(prod1)
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
    assert np.allclose(
        (squeeze(0) + displace(1)).to_matrix({
            0: 2,
            1: 2
        },
                                             displacement=0.5,
                                             squeezing=0.5),
        [[1.87758256, 0, -0.47942554, 0], [0, 1.87758256, 0, -0.47942554],
         [0.47942554, 0, 1.87758256, 0], [0, 0.47942554, 0, 1.87758256]])

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


def test_evaluation():
    displace_op = displace(1)
    squeeze_op = squeeze(3)
    coeff = ScalarOperator(lambda lam: 1. / lam)

    def evaluate(composite_op, **kwargs):
        return composite_op.evaluate(**kwargs)

    # test trivial evaluation
    get_op = lambda: create(0) + annihilate(1)
    assert numpy.allclose(get_op().to_matrix({
        0: 2,
        1: 3
    }),
                          get_op().evaluate().to_matrix({
                              0: 2,
                              1: 3
                          }))
    assert numpy.allclose(get_op().to_matrix({
        0: 2,
        1: 3
    }),
                          evaluate(get_op()).to_matrix({
                              0: 2,
                              1: 3
                          }))

    # test non-trivial evaluation
    def check_evaluation(composite_op):
        params1 = {
            "displacement": 0.05000126,
            "squeezing": 10.006008j,
            "lam": -0.51237 + 98.72035j
        }
        params2 = {
            "squeezing": 10.006008j,
            "displacement": 0.05000126,
            "lam": -0.51237 + 98.72035j
        }
        assert params1 == params2
        assert [e for e in params1] != [e for e in params2]

        # check that order of parameters does not matter
        eval1 = composite_op.evaluate(**params1)
        eval2 = composite_op.evaluate(**params2)
        assert len(eval1.parameters) == 0
        assert eval1 == eval2
        assert eval1 != composite_op.evaluate(squeezing=0.05000126,
                                              displacement=10.006008j,
                                              lam=-0.51237 + 98.72035j)
        assert eval1 == evaluate(composite_op, **params1)

        dims = {1: 3, 3: 4}
        assert numpy.allclose(eval1.to_matrix(dims), eval2.to_matrix(dims))
        params3 = {
            "displacement": 1.05000126,
            "squeezing": 10.006008j,
            "lam": -0.51237 + 98.72035j
        }
        eval3 = composite_op.evaluate(**params3)
        assert numpy.allclose(eval1.to_matrix(dims), eval2.to_matrix(dims))
        assert not numpy.allclose(eval1.to_matrix(dims), eval3.to_matrix(dims))

        # testing that we have a reasonable precision
        params4 = {
            "displacement": 1.05000126000000006,
            "squeezing": 10.006008j,
            "lam": -0.51237 + 98.72035j
        }
        assert params3 != params4
        eval4 = composite_op.evaluate(**params4)
        assert eval3 != eval4
        if type(composite_op) == MatrixOperatorTerm:
            for op1, op2 in zip(eval3, eval4):
                assert op1.id.startswith("displace") or op1.id.startswith(
                    "squeeze")
                if op1.id.startswith("squeeze"):
                    assert op1.id == op2.id
                else:
                    assert op1.id != op2.id

    check_evaluation(coeff * displace_op * squeeze_op)
    check_evaluation(coeff * displace_op + squeeze_op)


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


# made a separate function just to check that this works
def define_ops():

    def op_definition(dim):
        return np.diag([(-1. + 0j)**i for i in range(dim)])

    define("custom_parity1", [0],
           lambda dim: np.diag([(-1. + 0j)**i for i in range(dim)]))
    operators.define("custom_parity2", [0], op_definition)


def test_custom_operators():

    define("custom", [0],
           lambda dim: np.diag(np.zeros(dim, dtype=np.complex128)))
    with pytest.raises(Exception):
        # redefinition of an operator with the same name should raise and exception by default
        define("custom", [0],
               lambda dim: np.diag(np.zeros(dim, dtype=np.complex128)))
    define("custom", [0],
           lambda dim: np.diag(np.ones(dim, dtype=np.complex128)),
           override=True)
    custom1 = instantiate("custom", 2)
    assert np.allclose(custom1.to_matrix({2: 3}), identity_matrix(3))

    define_ops()
    custom2 = instantiate("custom_parity1", 1)
    assert np.allclose(custom2.to_matrix({1: 5}), parity_matrix(5))
    custom3 = instantiate("custom_parity2", [1])
    assert np.allclose(custom3.to_matrix({1: 5}), parity_matrix(5))

    def phase(angle: float):
        return np.array([[1, 0], [0, cmath.exp(1j * angle)]],
                        dtype=np.complex128)

    define("custom_phase", [2], phase)
    custom_op = instantiate("custom_phase", [1])
    with pytest.raises(Exception):
        custom_op.to_matrix()  # missing parameter
    assert np.allclose(custom_op.to_matrix(angle=np.pi),
                       np.array([[1, 0], [0, -1]]))

    # matrix evaluation for custom operators with multiple degrees

    def func0(dims):
        return np.kron(momentum_matrix(dims[1]), position_matrix(dims[0]))

    def func1(dims):
        return np.kron(momentum_matrix(dims[1]), number_matrix(dims[0]))

    define("custom_op0", [-1, -1], func0)
    define("custom_op1", [-1, -1], func1)
    dims = {0: 3, 1: 4, 2: 2, 3: 5}

    op0 = instantiate("custom_op0", [0, 1])
    op1 = instantiate("custom_op1", [1, 2])
    prod1 = op0 * op1
    prod2 = op1 * op0

    expected1 = np.dot(momentum_matrix(4), number_matrix(4))
    expected1 = np.kron(momentum_matrix(2), expected1)
    expected1 = np.kron(expected1, position_matrix(3))

    expected2 = np.dot(number_matrix(4), momentum_matrix(4))
    expected2 = np.kron(momentum_matrix(2), expected2)
    expected2 = np.kron(expected2, position_matrix(3))

    assert np.allclose(prod1.to_matrix(dims), expected1)
    assert np.allclose(prod2.to_matrix(dims), expected2)

    op0 = instantiate("custom_op0", [2, 3])
    op1 = instantiate("custom_op1", [0, 2])
    prod1 = op0 * op1
    prod2 = op1 * op0

    expected1 = np.dot(position_matrix(2), momentum_matrix(2))
    expected1 = np.kron(momentum_matrix(5), expected1)
    expected1 = np.kron(expected1, number_matrix(3))

    expected2 = np.dot(momentum_matrix(2), position_matrix(2))
    expected2 = np.kron(momentum_matrix(5), expected2)
    expected2 = np.kron(expected2, number_matrix(3))

    assert np.allclose(prod1.to_matrix(dims), expected1)
    assert np.allclose(prod2.to_matrix(dims), expected2)

    def func0(dims):
        return np.kron(momentum_matrix(dims[1]), position_matrix(dims[0]))

    def func1(dims):
        return np.kron(parity_matrix(dims[1]), number_matrix(dims[0]))

    operators.define("custom_op0", [-1, -1], func0, override=True)
    operators.define("custom_op1", [-1, -1], func1, override=True)

    op0 = instantiate("custom_op0", [0, 1])
    op1 = instantiate("custom_op1", [1, 2])
    matrix0 = np.kron(np.kron(identity_matrix(2), momentum_matrix(4)),
                      position_matrix(3))
    matrix1 = np.kron(np.kron(parity_matrix(2), number_matrix(4)),
                      identity_matrix(3))

    sum1 = op0 + op1
    sum2 = op1 + op0
    diff1 = op0 - op1
    diff2 = op1 - op0

    assert np.allclose(sum1.to_matrix(dims), matrix0 + matrix1)
    assert np.allclose(sum2.to_matrix(dims), matrix0 + matrix1)
    assert np.allclose(diff1.to_matrix(dims), matrix0 - matrix1)
    assert np.allclose(diff2.to_matrix(dims), matrix1 - matrix0)

    op0 = instantiate("custom_op0", [2, 3])
    op1 = instantiate("custom_op1", [0, 2])
    matrix0 = np.kron(np.kron(momentum_matrix(5), position_matrix(2)),
                      identity_matrix(3))
    matrix1 = np.kron(np.kron(identity_matrix(5), parity_matrix(2)),
                      number_matrix(3))

    sum1 = op0 + op1
    sum2 = op1 + op0
    diff1 = op0 - op1
    diff2 = op1 - op0

    assert np.allclose(sum1.to_matrix(dims), matrix0 + matrix1)
    assert np.allclose(sum2.to_matrix(dims), matrix0 + matrix1)
    assert np.allclose(diff1.to_matrix(dims), matrix0 - matrix1)
    assert np.allclose(diff2.to_matrix(dims), matrix1 - matrix0)


def test_parameter_docs():

    # built-in operators
    squeeze_op = squeeze(0)
    displace_op = displace(0)
    combined_op = squeeze_op + displace_op
    squeeze_amp_docs = "Amplitude of the squeezing operator. See also https://en.wikipedia.org/wiki/Squeeze_operator."
    displace_amp_docs = "Amplitude of the displacement operator. See also https://en.wikipedia.org/wiki/Displacement_operator."

    assert 'squeezing' in squeeze_op.parameters
    assert squeeze_op.parameters['squeezing'] == squeeze_amp_docs
    assert 'displacement' in displace_op.parameters
    assert displace_op.parameters['displacement'] == displace_amp_docs
    assert 'squeezing' in combined_op.parameters
    assert 'displacement' in combined_op.parameters
    assert combined_op.parameters['squeezing'] == squeeze_op.parameters[
        'squeezing']
    assert combined_op.parameters['displacement'] == displace_op.parameters[
        'displacement']

    # custom operators
    def rz(angle: float):
        """
        Single-qubit rotation about the Z axis.
        """
        return np.array(
            [[cmath.exp(-0.5j * angle), 0], [0, cmath.exp(0.5j * angle)]],
            dtype=np.complex128)

    define("rz", [2], rz)
    rz_op = instantiate("rz", 0)
    assert 'angle' in rz_op.parameters
    assert rz_op.parameters['angle'] == ''  # no parameter docs

    def phase(angle: float):
        """
        Returns:
        a matrix that applies a phase of exp(i * angle)
        Args:
        angle(float): exponent of the applied phase
        """
        return np.array([[1, 0], [0, cmath.exp(1j * angle)]],
                        dtype=np.complex128)

    define("phase", [2], phase)
    phase_op = instantiate("phase", [1])
    assert 'angle' in phase_op.parameters
    assert phase_op.parameters['angle'] == "exponent of the applied phase"

    combined1 = MatrixOperatorTerm(rz_op) * MatrixOperatorTerm(
        phase_op)  # FIXME: default to returning term op
    assert 'angle' in combined1.parameters
    assert combined1.parameters['angle'] == "exponent of the applied phase"
    combined2 = MatrixOperatorTerm(phase_op) * MatrixOperatorTerm(rz_op)
    assert 'angle' in combined2.parameters
    assert combined2.parameters['angle'] == "exponent of the applied phase"
    combined3 = MatrixOperatorTerm(rz_op) + MatrixOperatorTerm(phase_op)
    assert 'angle' in combined3.parameters
    assert combined3.parameters['angle'] == "exponent of the applied phase"
    combined4 = MatrixOperatorTerm(phase_op) - MatrixOperatorTerm(rz_op)
    assert 'angle' in combined4.parameters
    assert combined4.parameters['angle'] == "exponent of the applied phase"

    def phase2(angle: float):
        """Args:
        angle(float): different docs for angle...
        """
        return np.array([[1, 0], [0, cmath.exp(0.5j * angle)]],
                        dtype=np.complex128)

    operators.define("phase2", [2], phase2)
    phase_op2 = instantiate("phase2", 1)
    assert 'angle' in phase_op2.parameters
    assert phase_op2.parameters['angle'] == "different docs for angle..."

    # parameters are defined by their name - the same name means the same parameter
    # if multiple documentations exist for the same parameter, one description will
    # picked "at random" (depending on internal ordering of operator)
    combined5 = MatrixOperatorTerm(phase_op2) * MatrixOperatorTerm(phase_op)
    assert 'angle' in combined5.parameters
    assert combined5.parameters['angle'] == "exponent of the applied phase"
    combined6 = MatrixOperatorTerm(phase_op) * MatrixOperatorTerm(phase_op2)
    assert 'angle' in combined6.parameters
    assert combined6.parameters['angle'] == "different docs for angle..."


def test_backwards_compatibility():
    scalar = const(3)
    assert type(scalar) == ScalarOperator
    assert scalar.evaluate() == 3
    with pytest.raises(ValueError):
        const(lambda: 5)
    with pytest.raises(ValueError):
        const('c')

    def check_composite(op_create, matrix_create, coeff_val):
        op1 = op_create(5)
        assert type(op1) == MatrixOperatorTerm
        assert op1.degrees == [5]
        assert np.allclose(op1.to_matrix({5: 3}), matrix_create(3))
        op3 = op_create([1, 3, 5])
        assert op3.degrees == [1, 3, 5]
        assert op3.coefficient.evaluate() == coeff_val
        assert np.allclose(op3.to_matrix({1: 2, 3: 2, 5: 2}), matrix_create(8))
        for element in op3:
            assert len(element.degrees) == 1
            assert np.allclose(element.to_matrix({element.degrees[0]: 3}),
                               matrix_create(3))

    check_composite(zero, zero_matrix, 0)
    check_composite(identity, identity_matrix, 1)
    with pytest.warns(DeprecationWarning):
        create(5)
    with pytest.warns(DeprecationWarning):
        annihilate(5)


# Run with: pytest -rP
if __name__ == "__main__":
    pytest.main(["-rP"])
