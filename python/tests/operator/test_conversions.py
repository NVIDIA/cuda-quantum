# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, numpy as np, operator, pytest
from cudaq import boson, fermion, operators, spin
from op_utils import *  # test helpers


def test_product_conversions():
    params = {"squeezing": 0.5, "displacement": 0.25}
    dims = {0: 2, 1: 2}
    matrix_product = operators.squeeze(0) * operators.displace(1)
    matrix_product_expected = np.kron(displace_matrix(2, 0.25),
                                      squeeze_matrix(2, 0.5))
    spin_product = spin.y(1) * spin.x(0)
    spin_product_expected = np.kron(pauliy_matrix(), paulix_matrix())
    boson_product = boson.annihilate(1) * boson.number(0)
    boson_product_expected = np.kron(annihilate_matrix(2), number_matrix(2))
    # Combining fermion operators on the same targets with other operators
    # generally should result in an error being raised for conflicting commutation
    # relations. However, this error is only raised upon product creation, not
    # necessarily when creating the sum. (FIXME?)
    fermion_product = fermion.annihilate(0) * fermion.create(1)
    fermion_product_expected = np.kron(create_matrix(2), annihilate_matrix(2))

    product_ops = [(matrix_product, matrix_product_expected),
                   (spin_product, spin_product_expected),
                   (boson_product, boson_product_expected),
                   (fermion_product, fermion_product_expected)]
    sum_type = dict([(operators.MatrixOperatorTerm, operators.MatrixOperator),
                     (spin.SpinOperatorTerm, spin.SpinOperator),
                     (boson.BosonOperatorTerm, boson.BosonOperator),
                     (fermion.FermionOperatorTerm, fermion.FermionOperator)])

    def check_equals(op, expected, expected_type):
        assert type(op) == expected_type
        assert np.allclose(op.to_matrix(dims, params), expected)

    for (op1, expected1) in product_ops:
        for (op2, expected2) in product_ops:
            print("check arithmetics for {t1} with {t2}".format(t1=type(op1),
                                                                t2=type(op2)))
            base_type = type(op1)
            if type(op1) != type(op2):
                base_type = operators.MatrixOperatorTerm
            check_equals(op1 + op2, expected1 + expected2, sum_type[base_type])
            check_equals(op1 - op2, expected1 - expected2, sum_type[base_type])
            if isinstance(op1, fermion.FermionOperatorTerm) != isinstance(
                    op2, fermion.FermionOperatorTerm):
                with pytest.raises(Exception):
                    op1 * op2
            else:
                check_equals(op1 * op2, np.dot(expected1, expected2), base_type)
    for (op, expected) in product_ops:
        print(
            "check in-place arithmetics for {t} with itself".format(t=type(op)))
        prod = op.copy()
        prod *= op
        check_equals(prod, np.dot(expected, expected), type(op))
    for (op, expected) in product_ops[1:]:
        print("check in-place arithmetics for matrix product with {t}".format(
            t=type(op)))
        prod = matrix_product.copy()
        if isinstance(op, fermion.FermionOperatorTerm):
            with pytest.raises(Exception):
                prod *= op
        else:
            prod *= op
            assert type(prod) == operators.MatrixOperatorTerm
            check_equals(prod, np.dot(matrix_product_expected, expected),
                         operators.MatrixOperatorTerm)


def test_sum_conversions():
    params = {"squeezing": 0.5, "displacement": 0.25}
    dims = {0: 2, 1: 2}
    matrix_product = operators.squeeze(0) * operators.displace(1)
    matrix_product_expected = np.kron(displace_matrix(2, 0.25),
                                      squeeze_matrix(2, 0.5))
    spin_product = spin.y(1) * spin.x(0)
    spin_product_expected = np.kron(pauliy_matrix(), paulix_matrix())
    boson_product = boson.annihilate(1) * boson.number(0)
    boson_product_expected = np.kron(annihilate_matrix(2), number_matrix(2))
    # Combining fermion operators on the same targets with other operators
    # generally should result in an error being raised for conflicting commutation
    # relations. However, this error is only raised upon product creation, not
    # necessarily when creating the sum. (FIXME?)
    fermion_product = fermion.annihilate(0) * fermion.create(1)
    fermion_product_expected = np.kron(create_matrix(2), annihilate_matrix(2))

    product_ops = [(matrix_product, matrix_product_expected),
                   (spin_product, spin_product_expected),
                   (boson_product, boson_product_expected),
                   (fermion_product, fermion_product_expected)]

    matrix_sum = operators.squeeze(0) + operators.displace(1)
    matrix_sum_expected = np.kron(displace_matrix(2, 0.25), identity_matrix(2)) +\
                          np.kron(identity_matrix(2), squeeze_matrix(2, 0.5))
    spin_sum = spin.y(1) + spin.x(0)
    spin_sum_expected = np.kron(pauliy_matrix(), identity_matrix(2)) +\
                        np.kron(identity_matrix(2), paulix_matrix())
    boson_sum = boson.annihilate(1) + boson.number(0)
    boson_sum_expected = np.kron(annihilate_matrix(2), identity_matrix(2)) +\
                         np.kron(identity_matrix(2), number_matrix(2))
    fermion_sum = fermion.annihilate(0) + fermion.create(1)
    fermion_sum_expected = np.kron(create_matrix(2), identity_matrix(2)) +\
                           np.kron(identity_matrix(2), annihilate_matrix(2))

    sum_ops = [(matrix_sum, matrix_sum_expected), (spin_sum, spin_sum_expected),
               (boson_sum, boson_sum_expected),
               (fermion_sum, fermion_sum_expected)]

    sum_type = dict([(operators.MatrixOperatorTerm, operators.MatrixOperator),
                     (spin.SpinOperatorTerm, spin.SpinOperator),
                     (boson.BosonOperatorTerm, boson.BosonOperator),
                     (fermion.FermionOperatorTerm, fermion.FermionOperator)])

    def check_equals(op, expected, expected_type):
        assert type(op) == expected_type
        assert np.allclose(op.to_matrix(dims, params), expected)

    for (sum1, expected1) in sum_ops:
        for (sum2, expected2) in sum_ops:
            print("check arithmetics for {t1} with {t2}".format(t1=type(sum1),
                                                                t2=type(sum2)))
            expected_type = type(sum1)
            if type(sum1) != type(sum2):
                expected_type = operators.MatrixOperator
            check_equals(sum1 + sum2, expected1 + expected2, expected_type)
            check_equals(sum1 - sum2, expected1 - expected2, expected_type)
            if isinstance(sum1, fermion.FermionOperator) != isinstance(
                    sum2, fermion.FermionOperator):
                with pytest.raises(Exception):
                    sum1 * sum2
            elif isinstance(sum1, fermion.FermionOperator):
                # need to take commutation relations into account here...
                check_equals(sum1 * sum2, zero_matrix(4), expected_type)
            else:
                check_equals(sum1 * sum2, np.dot(expected1, expected2),
                             expected_type)

    for (sum, sum_expected) in sum_ops:
        for (prod, prod_expected) in product_ops:
            print("check arithmetics for {t1} with {t2} and vice versa".format(
                t1=type(sum), t2=type(prod)))
            expected_type = operators.MatrixOperator
            if sum_type[type(prod)] == type(sum):
                expected_type = type(sum)
            check_equals(sum + prod, sum_expected + prod_expected,
                         expected_type)
            check_equals(prod + sum, sum_expected + prod_expected,
                         expected_type)
            check_equals(sum - prod, sum_expected - prod_expected,
                         expected_type)
            check_equals(prod - sum, prod_expected - sum_expected,
                         expected_type)
            if isinstance(sum, fermion.FermionOperator) != isinstance(
                    prod, fermion.FermionOperatorTerm):
                with pytest.raises(Exception):
                    sum * prod
                with pytest.raises(Exception):
                    prod * sum
            else:
                check_equals(sum * prod, np.dot(sum_expected, prod_expected),
                             expected_type)
                check_equals(prod * sum, np.dot(prod_expected, sum_expected),
                             expected_type)

    for (op, expected) in sum_ops:
        print(
            "check in-place arithmetics for {t} with itself".format(t=type(op)))
        sum = op.copy()
        sum += op
        check_equals(sum, expected + expected, type(op))
        sum = op.copy()
        sum -= op
        check_equals(sum, zero_matrix(4), type(op))
        sum = op.copy()
        sum *= op
        if isinstance(sum, fermion.FermionOperator):
            # taking commutation relations into account
            check_equals(sum, zero_matrix(4), type(op))
        else:
            check_equals(sum, np.dot(expected, expected), type(op))
    for (op, expected) in sum_ops[1:]:
        print("check in-place arithmetics for matrix sum with {t}".format(
            t=type(op)))
        sum = matrix_sum.copy()
        sum += op
        check_equals(sum, matrix_sum_expected + expected,
                     operators.MatrixOperator)
        sum = matrix_sum.copy()
        sum -= op
        check_equals(sum, matrix_sum_expected - expected,
                     operators.MatrixOperator)
        sum = matrix_sum.copy()
        if isinstance(op, fermion.FermionOperator):
            with pytest.raises(Exception):
                sum *= op
        else:
            sum *= op
            check_equals(sum, np.dot(matrix_sum_expected, expected),
                         operators.MatrixOperator)
    for (op, expected) in product_ops[1:]:
        print("check in-place arithmetics for matrix sum with {t}".format(
            t=type(op)))
        sum = matrix_sum.copy()
        sum += op
        check_equals(sum, matrix_sum_expected + expected,
                     operators.MatrixOperator)
        sum = matrix_sum.copy()
        sum -= op
        check_equals(sum, matrix_sum_expected - expected,
                     operators.MatrixOperator)
        sum = matrix_sum.copy()
        if isinstance(op, fermion.FermionOperatorTerm):
            with pytest.raises(Exception):
                sum *= op
        else:
            sum *= op
            check_equals(sum, np.dot(matrix_sum_expected, expected),
                         operators.MatrixOperator)


def test_scalar_arithmetics():
    dims = {0: 2, 1: 2}
    scop = operators.const(2)
    assert type(scop) == cudaq.ScalarOperator
    for elop in (operators.identity(1), boson.identity(1), fermion.identity(1),
                 spin.i(1)):
        assert np.allclose((scop + elop).to_matrix(dims),
                           (elop + scop).to_matrix(dims))
        assert np.allclose((scop - elop).to_matrix(dims),
                           -(elop - scop).to_matrix(dims))
        assert np.allclose((scop * elop).to_matrix(dims),
                           (elop * scop).to_matrix(dims))
        assert np.allclose((elop / scop).to_matrix(dims), [[0.5, 0], [0, 0.5]])
        assert np.allclose(((scop * elop) / scop).to_matrix(dims),
                           [[1, 0], [0, 1]])
        assert np.allclose(((elop / scop) * elop).to_matrix(dims),
                           [[0.5, 0], [0, 0.5]])
        assert np.allclose(((elop / scop) + elop).to_matrix(dims),
                           [[1.5, 0], [0, 1.5]])
        assert np.allclose((elop * (elop / scop)).to_matrix(dims),
                           [[0.5, 0], [0, 0.5]])
        assert np.allclose((elop + (elop / scop)).to_matrix(dims),
                           [[1.5, 0], [0, 1.5]])
        assert np.allclose(((scop + elop) / scop).to_matrix(dims),
                           [[1.5, 0], [0, 1.5]])
        assert np.allclose((scop + (elop / scop)).to_matrix(dims),
                           [[2.5, 0], [0, 2.5]])
        assert np.allclose(((scop * elop) / scop).to_matrix(dims),
                           [[1, 0], [0, 1]])
        assert np.allclose((scop * (elop / scop)).to_matrix(dims),
                           [[1, 0], [0, 1]])
        assert np.allclose(((scop * elop) * scop).to_matrix(dims),
                           [[4, 0], [0, 4]])
        assert np.allclose((scop * (scop * elop)).to_matrix(dims),
                           [[4, 0], [0, 4]])
        assert np.allclose(((scop * elop) * elop).to_matrix(dims),
                           [[2, 0], [0, 2]])
        assert np.allclose((elop * (scop * elop)).to_matrix(dims),
                           [[2, 0], [0, 2]])
        assert np.allclose(((scop * elop) + scop).to_matrix(dims),
                           [[4, 0], [0, 4]])
        assert np.allclose((scop + (scop * elop)).to_matrix(dims),
                           [[4, 0], [0, 4]])
        assert np.allclose(((scop * elop) + elop).to_matrix(dims),
                           [[3, 0], [0, 3]])
        assert np.allclose((elop + (scop * elop)).to_matrix(dims),
                           [[3, 0], [0, 3]])
        assert np.allclose(((scop * elop) - scop).to_matrix(dims),
                           [[0, 0], [0, 0]])
        assert np.allclose((scop - (scop * elop)).to_matrix(dims),
                           [[0, 0], [0, 0]])
        assert np.allclose(((scop * elop) - elop).to_matrix(dims),
                           [[1, 0], [0, 1]])
        assert np.allclose((elop - (scop * elop)).to_matrix(dims),
                           [[-1, 0], [0, -1]])
        assert np.allclose(((scop + elop) * scop).to_matrix(dims),
                           [[6, 0], [0, 6]])
        assert np.allclose((scop * (scop + elop)).to_matrix(dims),
                           [[6, 0], [0, 6]])
        assert np.allclose(((scop + elop) * elop).to_matrix(dims),
                           [[3, 0], [0, 3]])
        assert np.allclose((elop * (scop + elop)).to_matrix(dims),
                           [[3, 0], [0, 3]])
        assert np.allclose(((scop - elop) * scop).to_matrix(dims),
                           [[2, 0], [0, 2]])
        assert np.allclose((scop * (scop - elop)).to_matrix(dims),
                           [[2, 0], [0, 2]])
        assert np.allclose(((scop - elop) * elop).to_matrix(dims),
                           [[1, 0], [0, 1]])
        assert np.allclose((elop * (scop - elop)).to_matrix(dims),
                           [[1, 0], [0, 1]])
        assert np.allclose(((scop + elop) + scop).to_matrix(dims),
                           [[5, 0], [0, 5]])
        assert np.allclose((scop + (scop + elop)).to_matrix(dims),
                           [[5, 0], [0, 5]])
        assert np.allclose(((scop + elop) + elop).to_matrix(dims),
                           [[4, 0], [0, 4]])
        assert np.allclose((elop + (scop + elop)).to_matrix(dims),
                           [[4, 0], [0, 4]])
        assert np.allclose(((scop - elop) - scop).to_matrix(dims),
                           [[-1, 0], [0, -1]])
        assert np.allclose((scop - (scop - elop)).to_matrix(dims),
                           [[1, 0], [0, 1]])
        assert np.allclose(((scop - elop) - elop).to_matrix(dims),
                           [[0, 0], [0, 0]])
        assert np.allclose((elop - (scop - elop)).to_matrix(dims),
                           [[0, 0], [0, 0]])

        assert np.allclose(
            operator.add(elop, 2).to_matrix(dims), [[3, 0], [0, 3]])
        assert np.allclose(
            operator.add(elop, 2.5).to_matrix(dims), [[3.5, 0], [0, 3.5]])
        assert np.allclose(
            operator.add(elop, 2j).to_matrix(dims), [[1 + 2j, 0], [0, 1 + 2j]])
        assert np.allclose(
            operator.add(2, elop).to_matrix(dims), [[3, 0], [0, 3]])
        assert np.allclose(
            operator.add(2.5, elop).to_matrix(dims), [[3.5, 0], [0, 3.5]])
        assert np.allclose(
            operator.add(2j, elop).to_matrix(dims), [[1 + 2j, 0], [0, 1 + 2j]])

        assert np.allclose(
            operator.sub(elop, 2).to_matrix(dims), [[-1, 0], [0, -1]])
        assert np.allclose(
            operator.sub(elop, 2.5).to_matrix(dims), [[-1.5, 0], [0, -1.5]])
        assert np.allclose(
            operator.sub(elop, 2j).to_matrix(dims), [[1 - 2j, 0], [0, 1 - 2j]])
        assert np.allclose(
            operator.sub(2, elop).to_matrix(dims), [[1, 0], [0, 1]])
        assert np.allclose(
            operator.sub(2.5, elop).to_matrix(dims), [[1.5, 0], [0, 1.5]])
        assert np.allclose(
            operator.sub(2j, elop).to_matrix(dims),
            [[-1 + 2j, 0], [0, -1 + 2j]])

        assert np.allclose(
            operator.mul(elop, 2).to_matrix(dims), [[2, 0], [0, 2]])
        assert np.allclose(
            operator.mul(elop, 2.5).to_matrix(dims), [[2.5, 0], [0, 2.5]])
        assert np.allclose(
            operator.mul(elop, 2j).to_matrix(dims), [[2j, 0], [0, 2j]])
        assert np.allclose(
            operator.mul(2, elop).to_matrix(dims), [[2, 0], [0, 2]])
        assert np.allclose(
            operator.mul(2.5, elop).to_matrix(dims), [[2.5, 0], [0, 2.5]])
        assert np.allclose(
            operator.mul(2j, elop).to_matrix(dims), [[2j, 0], [0, 2j]])

        assert np.allclose((elop / 2).to_matrix(dims), [[0.5, 0], [0, 0.5]])
        assert np.allclose((elop / 2.5).to_matrix(dims), [[0.4, 0], [0, 0.4]])
        assert np.allclose((elop / 2j).to_matrix(dims),
                           [[-0.5j, 0], [0, -0.5j]])

    operators.define("ops_create", [0], lambda dim: create_matrix(dim))
    operators.define("ops_annihilate", [0], lambda dim: annihilate_matrix(dim))
    custom_create = lambda target: operators.instantiate("ops_create", target)
    custom_annihilate = lambda target: operators.instantiate(
        "ops_annihilate", [target])

    for opprod in (
            custom_create(0) * custom_annihilate(0),
            boson.create(0) * boson.annihilate(0),
            fermion.create(0) * fermion.annihilate(0),
    ):
        assert np.allclose(
            operator.add(opprod, 2).to_matrix(dims), [[2, 0], [0, 3]])
        assert np.allclose(
            operator.add(opprod, 2.5).to_matrix(dims), [[2.5, 0], [0, 3.5]])
        assert np.allclose(
            operator.add(opprod, 2j).to_matrix(dims), [[2j, 0], [0, 1 + 2j]])
        assert np.allclose(
            operator.add(2, opprod).to_matrix(dims), [[2, 0], [0, 3]])
        assert np.allclose(
            operator.add(2.5, opprod).to_matrix(dims), [[2.5, 0], [0, 3.5]])
        assert np.allclose(
            operator.add(2j, opprod).to_matrix(dims), [[2j, 0], [0, 1 + 2j]])

        assert np.allclose(
            operator.sub(opprod, 2).to_matrix(dims), [[-2, 0], [0, -1]])
        assert np.allclose(
            operator.sub(opprod, 2.5).to_matrix(dims), [[-2.5, 0], [0, -1.5]])
        assert np.allclose(
            operator.sub(opprod, 2j).to_matrix(dims), [[-2j, 0], [0, 1 - 2j]])
        assert np.allclose(
            operator.sub(2, opprod).to_matrix(dims), [[2, 0], [0, 1]])
        assert np.allclose(
            operator.sub(2.5, opprod).to_matrix(dims), [[2.5, 0], [0, 1.5]])
        assert np.allclose(
            operator.sub(2j, opprod).to_matrix(dims), [[2j, 0], [0, -1 + 2j]])

        assert np.allclose(
            operator.mul(opprod, 2).to_matrix(dims), [[0, 0], [0, 2]])
        assert np.allclose(
            operator.mul(opprod, 2.5).to_matrix(dims), [[0, 0], [0, 2.5]])
        assert np.allclose(
            operator.mul(opprod, 2j).to_matrix(dims), [[0, 0], [0, 2j]])
        assert np.allclose(
            operator.mul(2, opprod).to_matrix(dims), [[0, 0], [0, 2]])
        assert np.allclose(
            operator.mul(2.5, opprod).to_matrix(dims), [[0, 0], [0, 2.5]])
        assert np.allclose(
            operator.mul(2j, opprod).to_matrix(dims), [[0, 0], [0, 2j]])

        assert np.allclose((opprod / 2).to_matrix(dims), [[0, 0], [0, 0.5]])
        assert np.allclose((opprod / 2.5).to_matrix(dims), [[0, 0], [0, 0.4]])
        assert np.allclose((opprod / 2j).to_matrix(dims), [[0, 0], [0, -0.5j]])

    for opsum in (
            custom_create(0) + custom_annihilate(0),
            boson.create(0) + boson.annihilate(0),
            fermion.create(0) + fermion.annihilate(0),
            spin.minus(0) + spin.plus(0),
    ):
        assert np.allclose(
            operator.add(opsum, 2).to_matrix(dims), [[2, 1], [1, 2]])
        assert np.allclose(
            operator.add(opsum, 2.5).to_matrix(dims), [[2.5, 1], [1, 2.5]])
        assert np.allclose(
            operator.add(opsum, 2j).to_matrix(dims), [[2j, 1], [1, 2j]])
        assert np.allclose(
            operator.add(2, opsum).to_matrix(dims), [[2, 1], [1, 2]])
        assert np.allclose(
            operator.add(2.5, opsum).to_matrix(dims), [[2.5, 1], [1, 2.5]])
        assert np.allclose(
            operator.add(2j, opsum).to_matrix(dims), [[2j, 1], [1, 2j]])

        assert np.allclose(
            operator.sub(opsum, 2).to_matrix(dims), [[-2, 1], [1, -2]])
        assert np.allclose(
            operator.sub(opsum, 2.5).to_matrix(dims), [[-2.5, 1], [1, -2.5]])
        assert np.allclose(
            operator.sub(opsum, 2j).to_matrix(dims), [[-2j, 1], [1, -2j]])
        assert np.allclose(
            operator.sub(2, opsum).to_matrix(dims), [[2, -1], [-1, 2]])
        assert np.allclose(
            operator.sub(2.5, opsum).to_matrix(dims), [[2.5, -1], [-1, 2.5]])
        assert np.allclose(
            operator.sub(2j, opsum).to_matrix(dims), [[2j, -1], [-1, 2j]])

        assert np.allclose(
            operator.mul(opsum, 2).to_matrix(dims), [[0, 2], [2, 0]])
        assert np.allclose(
            operator.mul(opsum, 2.5).to_matrix(dims), [[0, 2.5], [2.5, 0]])
        assert np.allclose(
            operator.mul(opsum, 2j).to_matrix(dims), [[0, 2j], [2j, 0]])
        assert np.allclose(
            operator.mul(2, opsum).to_matrix(dims), [[0, 2], [2, 0]])
        assert np.allclose(
            operator.mul(2.5, opsum).to_matrix(dims), [[0, 2.5], [2.5, 0]])
        assert np.allclose(
            operator.mul(2j, opsum).to_matrix(dims), [[0, 2j], [2j, 0]])

        assert np.allclose((opsum / 2).to_matrix(dims), [[0, 0.5], [0.5, 0]])
        assert np.allclose((opsum / 2.5).to_matrix(dims), [[0, 0.4], [0.4, 0]])
        assert np.allclose((opsum / 2j).to_matrix(dims),
                           [[0, -0.5j], [-0.5j, 0]])


def test_equality():
    op_sum = operators.squeeze(0) + operators.displace(1)
    op_prod = operators.squeeze(0) * operators.displace(1)
    boson_sum = boson.create(0) + boson.annihilate(1)
    boson_prod = boson.create(0) * boson.annihilate(1)
    fermion_sum = fermion.annihilate(0) + fermion.create(1)
    fermion_prod = fermion.annihilate(0) * fermion.create(1)
    spin_sum = spin.x(0) + spin.z(1)
    spin_prod = spin.x(0) * spin.z(1)
    ops = [
        op_sum,
        op_prod,
        boson_sum,
        boson_prod,
        fermion_sum,
        fermion_prod,
        spin_sum,
        spin_prod,
    ]
    for op in ops:
        for other in (spin.y, boson.create, fermion.number):
            assert (op * other(2)) == (other(2) * op)
            assert (op * other(2)) != (other(3) * op)
            assert (op + other(0)) == (other(0) + op)
            assert (op + other(0)) != (other(1) + op)


# Run with: pytest -rP
if __name__ == "__main__":
    pytest.main(["-rP"])
