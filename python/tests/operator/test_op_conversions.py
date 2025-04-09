# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np, pytest
from cudaq import boson, fermion, ops, spin_op # FIXME
from op_utils import * # test helpers


def test_product_conversions():
    params = {"squeezing": 0.5, "displacement": 0.25}
    dims = {0: 2, 1: 2}
    matrix_product = ops.squeeze(0) * ops.displace(1)
    matrix_product_expected = np.kron(displace_matrix(2, 0.25), squeeze_matrix(2, 0.5))
    spin_product = spin_op.y(1) * spin_op.x(0)
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
    sum_type = dict([(ops.MatrixOperatorTerm, ops.MatrixOperator), 
                     (spin_op.SpinOperatorTerm, spin_op.SpinOperator), 
                     (boson.BosonOperatorTerm, boson.BosonOperator), 
                     (fermion.FermionOperatorTerm, fermion.FermionOperator)])

    def check_equals(op, expected, expected_type):
        assert type(op) == expected_type
        assert np.allclose(op.to_matrix(dims, params), expected)

    for (op1, expected1) in product_ops:
        for (op2, expected2) in product_ops:
            print("check arithmetics for {t1} with {t2}".format(t1=type(op1), t2=type(op2)))
            base_type = type(op1)
            if type(op1) != type(op2):
                base_type = ops.MatrixOperatorTerm
            check_equals(op1 + op2, expected1 + expected2, sum_type[base_type])
            check_equals(op1 - op2, expected1 - expected2, sum_type[base_type])
            if isinstance(op1, fermion.FermionOperatorTerm) != isinstance(op2, fermion.FermionOperatorTerm):
                with pytest.raises(Exception): op1 * op2 
            else:
                check_equals(op1 * op2, np.dot(expected1, expected2), base_type)
    for (op, expected) in product_ops:
        print("check in-place arithmetics for {t} with itself".format(t=type(op)))
        prod = op.copy()
        prod *= op
        check_equals(prod, np.dot(expected, expected), type(op))
    for (op, expected) in product_ops[1:]:
        print("check in-place arithmetics for matrix product with {t}".format(t=type(op)))
        prod = matrix_product.copy()
        if isinstance(op, fermion.FermionOperatorTerm):
            with pytest.raises(Exception): prod *= op
        else:
            prod *= op
            assert type(prod) == ops.MatrixOperatorTerm
            check_equals(prod, np.dot(matrix_product_expected, expected), ops.MatrixOperatorTerm)
    
    # FIXME: option to remap degrees of operator and test fermions without exception


def test_sum_conversions():
    params = {"squeezing": 0.5, "displacement": 0.25}
    dims = {0: 2, 1: 2}
    matrix_product = ops.squeeze(0) * ops.displace(1)
    matrix_product_expected = np.kron(displace_matrix(2, 0.25), squeeze_matrix(2, 0.5))
    spin_product = spin_op.y(1) * spin_op.x(0)
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

    matrix_sum = ops.squeeze(0) + ops.displace(1)
    matrix_sum_expected = np.kron(displace_matrix(2, 0.25), identity_matrix(2)) +\
                          np.kron(identity_matrix(2), squeeze_matrix(2, 0.5))
    spin_sum = spin_op.y(1) + spin_op.x(0)
    spin_sum_expected = np.kron(pauliy_matrix(), identity_matrix(2)) +\
                        np.kron(identity_matrix(2), paulix_matrix())
    boson_sum = boson.annihilate(1) + boson.number(0)
    boson_sum_expected = np.kron(annihilate_matrix(2), identity_matrix(2)) +\
                         np.kron(identity_matrix(2), number_matrix(2))
    fermion_sum = fermion.annihilate(0) + fermion.create(1)
    fermion_sum_expected = np.kron(create_matrix(2), identity_matrix(2)) +\
                           np.kron(identity_matrix(2), annihilate_matrix(2))

    sum_ops = [(matrix_sum, matrix_sum_expected), 
               (spin_sum, spin_sum_expected), 
               (boson_sum, boson_sum_expected),
               (fermion_sum, fermion_sum_expected)]

    sum_type = dict([(ops.MatrixOperatorTerm, ops.MatrixOperator), 
                     (spin_op.SpinOperatorTerm, spin_op.SpinOperator), 
                     (boson.BosonOperatorTerm, boson.BosonOperator), 
                     (fermion.FermionOperatorTerm, fermion.FermionOperator)])

    def check_equals(op, expected, expected_type):
        assert type(op) == expected_type
        assert np.allclose(op.to_matrix(dims, params), expected)

    for (sum1, expected1) in sum_ops:
        for (sum2, expected2) in sum_ops:
            print("check arithmetics for {t1} with {t2}".format(t1=type(sum1), t2=type(sum2)))
            expected_type = type(sum1)
            if type(sum1) != type(sum2):
                expected_type = ops.MatrixOperator
            check_equals(sum1 + sum2, expected1 + expected2, expected_type)
            check_equals(sum1 - sum2, expected1 - expected2, expected_type)
            if isinstance(sum1, fermion.FermionOperator) != isinstance(sum2, fermion.FermionOperator):
                with pytest.raises(Exception): sum1 * sum2
            elif isinstance(sum1, fermion.FermionOperator):
                # need to take commutation relations into account here...
                check_equals(sum1 * sum2, zero_matrix(4), expected_type)
            else:
                check_equals(sum1 * sum2, np.dot(expected1, expected2), expected_type)

    for (sum, sum_expected) in sum_ops:
        for (prod, prod_expected) in product_ops:
            print("check arithmetics for {t1} with {t2} and vice versa".format(t1=type(sum), t2=type(prod)))
            expected_type = ops.MatrixOperator
            if sum_type[type(prod)] == type(sum):
                expected_type = type(sum)
            check_equals(sum + prod, sum_expected + prod_expected, expected_type)
            check_equals(prod + sum, sum_expected + prod_expected, expected_type)
            check_equals(sum - prod, sum_expected - prod_expected, expected_type)
            check_equals(prod - sum, prod_expected - sum_expected, expected_type)
            if isinstance(sum, fermion.FermionOperator) != isinstance(prod, fermion.FermionOperatorTerm):
                with pytest.raises(Exception): sum * prod
                with pytest.raises(Exception): prod * sum
            else:
                check_equals(sum * prod, np.dot(sum_expected, prod_expected), expected_type)
                check_equals(prod * sum, np.dot(prod_expected, sum_expected), expected_type)

    for (op, expected) in sum_ops:
        print("check in-place arithmetics for {t} with itself".format(t=type(op)))
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
        print("check in-place arithmetics for matrix sum with {t}".format(t=type(op)))
        sum = matrix_sum.copy()
        sum += op
        check_equals(sum, matrix_sum_expected + expected, ops.MatrixOperator)
        sum = matrix_sum.copy()
        sum -= op
        check_equals(sum, matrix_sum_expected - expected, ops.MatrixOperator)
        sum = matrix_sum.copy()
        if isinstance(op, fermion.FermionOperator):
            with pytest.raises(Exception): sum *= op
        else:
            sum *= op
            check_equals(sum, np.dot(matrix_sum_expected, expected), ops.MatrixOperator)
    for (op, expected) in product_ops[1:]:
        print("check in-place arithmetics for matrix sum with {t}".format(t=type(op)))
        sum = matrix_sum.copy()
        sum += op
        check_equals(sum, matrix_sum_expected + expected, ops.MatrixOperator)
        sum = matrix_sum.copy()
        sum -= op
        check_equals(sum, matrix_sum_expected - expected, ops.MatrixOperator)
        sum = matrix_sum.copy()
        if isinstance(op, fermion.FermionOperatorTerm):
            with pytest.raises(Exception): sum *= op
        else:
            sum *= op
            check_equals(sum, np.dot(matrix_sum_expected, expected), ops.MatrixOperator)

    # FIXME: option to remap degrees of operator and test fermions without exception


# Run with: pytest -rP
if __name__ == "__main__":
    pytest.main(["-rP"])