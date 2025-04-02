# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, numpy as np, operator, pytest
from cudaq.operator import *


@pytest.fixture(autouse=True)
def setup_and_teardown():
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def test_pauli_matrices():
    dims = {0: 2, 1: 2, 2: 2}
    assert np.allclose(spin.x(1).to_matrix(dims), [[0, 1], [1, 0]])
    assert np.allclose(spin.y(2).to_matrix(dims), [[0, -1j], [1j, 0]])


def test_matrix_multiplication():
    dims = {0: 2, 1: 2}
    zz_00 = spin.z(0) * spin.z(0)
    zz_01 = spin.z(0) * spin.z(1)
    zy_01 = spin.z(0) * spin.y(1)
    assert np.allclose(zz_00.to_matrix(dims), np.eye(2))
    assert np.allclose(
        zz_01.to_matrix(dims),
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    assert np.allclose(
        zy_01.to_matrix(dims),
        [[0, 0, -1j, 0], [0, 0, 0, 1j], [1j, 0, 0, 0], [0, -1j, 0, 0]])

    dims = {0: 2}

    I = spin.i(0)
    X = spin.x(0)
    Y = spin.y(0)
    Z = spin.z(0)

    expected_results = {
        ("I", "I"): I,
        ("I", "X"): X,
        ("I", "Y"): Y,
        ("I", "Z"): Z,
        ("X", "I"): X,
        ("X", "X"): I,
        ("X", "Y"): 1j * Z,
        ("X", "Z"): -1j * Y,
        ("Y", "I"): Y,
        ("Y", "X"): -1j * Z,
        ("Y", "Y"): I,
        ("Y", "Z"): 1j * X,
        ("Z", "I"): Z,
        ("Z", "X"): 1j * Y,
        ("Z", "Y"): -1j * X,
        ("Z", "Z"): I,
    }

    for (op1_str, op2_str), expected in expected_results.items():
        op1 = locals()[op1_str]
        op2 = locals()[op2_str]
        product = op1 * op2
        expected_matrix = expected.to_matrix(dims)
        product_matrix = product.to_matrix(dims)
        assert np.allclose(product_matrix,
                           expected_matrix), f"Failed for {op1} * {op2}"


def test_operator_addition():
    dims = {0: 2, 1: 2}
    op1 = ProductOperator([spin.x(0), spin.i(1)])
    op2 = ProductOperator([spin.i(0), spin.x(1)])
    assert np.allclose((op1 + op2).to_matrix(dims),
                       [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
    assert np.allclose((op2 + op1).to_matrix(dims),
                       [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])


def test_complex_operations():
    dims = {0: 2, 1: 2}
    op1 = spin.x(0) * spin.x(1)
    op2 = spin.z(0) * spin.z(1)
    assert np.allclose(
        (op1 + op2).to_matrix(dims),
        [[1, 0, 0, 1], [0, -1, 1, 0], [0, 1, -1, 0], [1, 0, 0, 1]])
    op3 = spin.x(0) + spin.x(1)
    op4 = spin.z(0) + spin.z(1)
    assert np.allclose(
        (op3 * op4).to_matrix(dims),
        [[0, 0, 0, 0], [2, 0, 0, -2], [2, 0, 0, -2], [0, 0, 0, 0]])
    dims = {0: 2, 1: 2, 2: 2}
    op5 = operators.squeeze(0) + operators.displace(1)
    assert np.allclose(
        op5.to_matrix(dims, displacement=0.5, squeezing=0.5),
        [[1.87758256, 0, -0.47942554, 0], [0, 1.87758256, 0, -0.47942554],
         [0.47942554, 0, 1.87758256, 0], [0, 0.47942554, 0, 1.87758256]])
    assert np.allclose((op5 * spin.y(2)).to_matrix(dims,
                                                   displacement=0.5,
                                                   squeezing=0.5),
                       (spin.y(2) * op5).to_matrix(dims,
                                                   displacement=0.5,
                                                   squeezing=0.5))


def test_scalar_operations():
    dims = {0: 2}
    so1 = ScalarOperator(lambda t: t)
    assert so1.to_matrix(t=2.0) == 2.0
    op1 = so1 * spin.x(0)
    op2 = spin.x(0) * so1
    op3 = so1 + spin.x(0)
    op4 = spin.x(0) + so1
    assert np.allclose(op1.to_matrix(dims, t=2.0), [[0, 2], [2, 0]])
    assert np.allclose(op2.to_matrix(dims, t=2.0), [[0, 2], [2, 0]])
    assert np.allclose(op1.to_matrix(dims, t=2.0), op2.to_matrix(dims, t=2.0))
    assert np.allclose(op3.to_matrix(dims, t=2.0), [[2, 1], [1, 2]])
    assert np.allclose(op4.to_matrix(dims, t=2.0), [[2, 1], [1, 2]])
    assert np.allclose(op3.to_matrix(dims, t=2.0), op4.to_matrix(dims, t=2.0))
    assert np.allclose(op1.to_matrix(dims, t=1j), [[0, 1j], [1j, 0]])
    assert np.allclose(op2.to_matrix(dims, t=1j), [[0, 1j], [1j, 0]])
    assert np.allclose(op1.to_matrix(dims, t=2.0), op2.to_matrix(dims, t=2.0))
    assert np.allclose(op3.to_matrix(dims, t=1j), [[1j, 1], [1, 1j]])
    assert np.allclose(op4.to_matrix(dims, t=1j), [[1j, 1], [1, 1j]])
    assert np.allclose(op3.to_matrix(dims, t=2.0), op4.to_matrix(dims, t=2.0))


def test_scalar_constant():
    so_const = ScalarOperator.const(5)
    assert np.allclose(so_const.to_matrix(), [5])


def test_scalar_arithmetic_operations():
    so1 = ScalarOperator.const(3)
    so2 = ScalarOperator.const(2)

    assert np.allclose((so1 + so2).to_matrix(), [5])
    assert np.allclose((so1 - so2).to_matrix(), [1])
    assert np.allclose((so1 * so2).to_matrix(), [6])
    assert np.allclose((so1 / so2).to_matrix(), [1.5])
    assert np.allclose((so1**2).to_matrix(), [9])


def test_scalar_generator_update():
    so1 = ScalarOperator(lambda t: t)
    so1.generator = lambda p: 1 / p
    assert 'p' in so1.parameters

    dims = {0: 2}
    op = so1 * spin.x(0)
    assert np.allclose(op.to_matrix(dims, p=2.0), [[0, 0.5], [0.5, 0]])

    # Update to new generator function
    so1.generator = lambda q: q + 1
    assert 'q' in so1.parameters


def test_composite_scalar_operator_with_same_parameter():
    so1 = ScalarOperator(lambda t: t)
    so2 = ScalarOperator(lambda t: 2 * t)
    composite_op = so1 * so2

    dims = {0: 2}
    op = composite_op * spin.x(0)
    assert np.allclose(op.to_matrix(dims, t=2.0), [[0, 8], [8, 0]])


def test_composite_scalar_operator_with_different_parameter():
    so1 = ScalarOperator(lambda t: t)
    so2 = ScalarOperator(lambda p: 2 * p)
    composite_op = so1 * so2

    dims = {0: 2}
    op = composite_op * spin.x(0)
    assert np.allclose(op.to_matrix(dims, t=2.0, p=3.0), [[0, 12], [12, 0]])
    assert 't' in composite_op.parameters and 'p' in composite_op.parameters


def test_parameter_update_in_composite_operator():
    so1 = ScalarOperator(lambda t: t)
    so2 = ScalarOperator(lambda p: 2 * p)
    composite_op = so1 * so2

    so1.generator = lambda q: 1 / q
    assert 'q' in composite_op.parameters
    assert 'p' in composite_op.parameters


def test_scalar_operator_with_parameter_doc():
    # Define a generator with a parameter docstring
    def generator_with_doc(t):
        """Time parameter for scaling"""
        return t

    so = ScalarOperator(generator_with_doc)
    assert 't' in so.parameters


def test_composite_scalar_operator_with_parameter_doc():
    # Define generators with individual parameter docstrings
    def generator1(t):
        """Time parameter for first scalar operator"""
        return t

    def generator2(p):
        """Amplitude parameter for second scalar operator"""
        return p * 2

    so1 = ScalarOperator(generator1)
    so2 = ScalarOperator(generator2)

    composite_op = so1 * so2
    assert 't' in composite_op.parameters
    assert 'p' in composite_op.parameters


def test_update_parameter_doc_in_composite_operator():
    # Define initial generators with a docstring
    def generator1(t):
        """Initial time parameter"""
        return t

    def generator2(p):
        """Amplitude parameter"""
        return p * 2

    so1 = ScalarOperator(generator1)
    so2 = ScalarOperator(generator2)

    composite_op = so1 * so2
    assert 't' in composite_op.parameters
    assert 'p' in composite_op.parameters

    def new_generator(q):
        """New frequency parameter"""
        return 1 / q

    so1.generator = new_generator

    assert 'q' in composite_op.parameters
    assert 'p' in composite_op.parameters


def test_parameter_description():
    squeeze_op = operators.squeeze(0)
    displace_op = operators.displace(0)
    combined_op = squeeze_op + displace_op
    assert 'squeezing' in squeeze_op.parameters and squeeze_op.parameters[
        'squeezing'].strip()
    assert 'displacement' in displace_op.parameters and displace_op.parameters[
        'displacement'].strip()
    assert 'squeezing' in combined_op.parameters and combined_op.parameters[
        'squeezing'].strip()
    assert 'displacement' in combined_op.parameters and combined_op.parameters[
        'displacement'].strip()
    assert combined_op.parameters['squeezing'] == squeeze_op.parameters[
        'squeezing']
    assert combined_op.parameters['displacement'] == displace_op.parameters[
        'displacement']

    def generator(param1, args):
        """Some args documentation.
        Args:

        param1 (:obj:`int`, optional): my docs for param1
        args: Description of `args`. Multiple
                lines are supported.
        Returns:
        Something that depends on param1.
        """
        if param1:
            return 0
        else:
            return args

    op = ScalarOperator(generator)
    assert 'param1' in op.parameters and op.parameters[
        'param1'] == 'my docs for param1'
    assert 'args' in op.parameters and op.parameters[
        'args'] == 'Description of `args`. Multiple lines are supported.'


def test_operator_equality():
    # ScalarOperator constants and generators
    assert ScalarOperator.const(5) == ScalarOperator.const(5)
    assert ScalarOperator.const(5) == ScalarOperator.const(5 + 0j)
    assert ScalarOperator.const(5) != ScalarOperator.const(5j)
    assert ScalarOperator.const(lambda: 5) != ScalarOperator.const(5)
    assert ScalarOperator.const(lambda: 5) != ScalarOperator.const(lambda: 5)

    generator = lambda: 5
    so1 = ScalarOperator(generator)
    so2 = ScalarOperator(lambda: 5)

    assert so1 != so2
    assert so1 == ScalarOperator(generator)
    so2.generator = generator
    assert so1 == so2

    # Identity and elementary operators with scalar operators
    elop = ElementaryOperator.identity(1)
    assert (elop * so1) == (elop * so2)
    assert (elop + so1) == (elop + so2)

    # Spin operators
    assert (spin.x(1) + spin.y(1)) == ((spin.y(1) + spin.x(1)))
    assert (spin.x(1) * spin.y(1)) == ((spin.y(1) * spin.x(1)))
    assert (spin.x(0) + spin.y(1)) == ((spin.y(1) + spin.x(0)))
    assert (spin.x(0) * spin.y(1)) == ((spin.y(1) * spin.x(0)))

    # Product and sum of operators
    opprod = operators.create(0) * operators.annihilate(0)
    oppsum = operators.create(0) + operators.annihilate(0)
    assert opprod != oppsum
    assert (opprod * so1) == (so1 * opprod)
    assert (opprod + so1) == (so1 + opprod)

    # Pauli matrices
    spinzy = lambda i, j: spin.z(i) * spin.y(j)
    spinxy = lambda i, j: spin.x(i) * spin.y(j)
    assert (spinxy(0, 0) + spinzy(0, 0)) == (spinzy(0, 0) + spinxy(0, 0))
    assert (spinxy(0, 0) * spinzy(0, 0)) == (spinzy(0, 0) * spinxy(0, 0))
    assert (spinxy(1, 1) * spinzy(0, 0)) == (spinzy(0, 0) * spinxy(1, 1))
    assert (spinxy(1, 2) * spinzy(3, 4)) == (spinzy(3, 4) * spinxy(1, 2))

    # Scalar arithmetic and operator interactions
    assert (ScalarOperator.const(5) +
            ScalarOperator.const(3)) == (ScalarOperator.const(4) +
                                         ScalarOperator.const(4))
    assert (ScalarOperator.const(6) *
            ScalarOperator.const(2)) == (ScalarOperator.const(4) *
                                         ScalarOperator.const(3))
    assert ((ScalarOperator.const(5) + ScalarOperator.const(3)) *
            elop) == (elop *
                      (ScalarOperator.const(4) + ScalarOperator.const(4)))
    assert (ScalarOperator.const(6) * ScalarOperator.const(2) +
            elop) == (elop + ScalarOperator.const(4) * ScalarOperator.const(3))
    assert (ScalarOperator.const(6) * ScalarOperator.const(2) *
            elop) == (elop * ScalarOperator.const(4) * ScalarOperator.const(3))

    # Mixed scalar and operator arithmetic
    assert (ScalarOperator.const(5) + 3) == (4 + ScalarOperator.const(4))
    assert (ScalarOperator.const(6) * 2) == (4 * ScalarOperator.const(3))
    assert ((ScalarOperator.const(5) + 3) *
            elop) == (elop * (4 + ScalarOperator.const(4)))
    assert (ScalarOperator.const(6) * 2 + elop) == (elop +
                                                    4 * ScalarOperator.const(3))
    assert (ScalarOperator.const(6) * 2.0 * elop) == (elop * 4.0 *
                                                      ScalarOperator.const(3))
    assert (ScalarOperator.const(6) / 2) == ScalarOperator.const(3)


def test_arithmetics_operations():
    dims = {0: 2, 1: 2}
    scop = operators.const(2)
    elop = operators.identity(1)
    assert np.allclose((scop + elop).to_matrix(dims),
                       (elop + scop).to_matrix(dims))
    assert np.allclose((scop - elop).to_matrix(dims),
                       -(elop - scop).to_matrix(dims))
    assert np.allclose((scop * elop).to_matrix(dims),
                       (elop * scop).to_matrix(dims))
    assert np.allclose(
        (elop / scop).to_matrix(dims),
        [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]])
    assert np.allclose(((scop * elop) / scop).to_matrix(dims),
                       [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.allclose(
        ((elop / scop) * elop).to_matrix(dims),
        [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]])
    assert np.allclose(
        ((elop / scop) + elop).to_matrix(dims),
        [[1.5, 0, 0, 0], [0, 1.5, 0, 0], [0, 0, 1.5, 0], [0, 0, 0, 1.5]])
    assert np.allclose(
        (elop * (elop / scop)).to_matrix(dims),
        [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]])
    assert np.allclose(
        (elop + (elop / scop)).to_matrix(dims),
        [[1.5, 0, 0, 0], [0, 1.5, 0, 0], [0, 0, 1.5, 0], [0, 0, 0, 1.5]])
    assert np.allclose(
        ((scop + elop) / scop).to_matrix(dims),
        [[1.5, 0, 0, 0], [0, 1.5, 0, 0], [0, 0, 1.5, 0], [0, 0, 0, 1.5]])
    assert np.allclose(
        (scop + (elop / scop)).to_matrix(dims),
        [[2.5, 0, 0, 0], [0, 2.5, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 2.5]])
    assert np.allclose(((scop * elop) / scop).to_matrix(dims),
                       [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.allclose((scop * (elop / scop)).to_matrix(dims),
                       [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.allclose(((scop * elop) * scop).to_matrix(dims),
                       [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]])
    assert np.allclose((scop * (scop * elop)).to_matrix(dims),
                       [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]])
    assert np.allclose(((scop * elop) * elop).to_matrix(dims),
                       [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    assert np.allclose((elop * (scop * elop)).to_matrix(dims),
                       [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    assert np.allclose(((scop * elop) + scop).to_matrix(dims),
                       [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]])
    assert np.allclose((scop + (scop * elop)).to_matrix(dims),
                       [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]])
    assert np.allclose(((scop * elop) + elop).to_matrix(dims),
                       [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]])
    assert np.allclose((elop + (scop * elop)).to_matrix(dims),
                       [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]])
    assert np.allclose(((scop * elop) - scop).to_matrix(dims),
                       [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert np.allclose((scop - (scop * elop)).to_matrix(dims),
                       [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert np.allclose(((scop * elop) - elop).to_matrix(dims),
                       [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.allclose(
        (elop - (scop * elop)).to_matrix(dims),
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    assert np.allclose(((scop + elop) * scop).to_matrix(dims),
                       [[6, 0, 0, 0], [0, 6, 0, 0], [0, 0, 6, 0], [0, 0, 0, 6]])
    assert np.allclose((scop * (scop + elop)).to_matrix(dims),
                       [[6, 0, 0, 0], [0, 6, 0, 0], [0, 0, 6, 0], [0, 0, 0, 6]])
    assert np.allclose(((scop + elop) * elop).to_matrix(dims),
                       [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]])
    assert np.allclose((elop * (scop + elop)).to_matrix(dims),
                       [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]])
    assert np.allclose(((scop - elop) * scop).to_matrix(dims),
                       [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    assert np.allclose((scop * (scop - elop)).to_matrix(dims),
                       [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    assert np.allclose(((scop - elop) * elop).to_matrix(dims),
                       [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.allclose((elop * (scop - elop)).to_matrix(dims),
                       [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.allclose(((scop + elop) + scop).to_matrix(dims),
                       [[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]])
    assert np.allclose((scop + (scop + elop)).to_matrix(dims),
                       [[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]])
    assert np.allclose(((scop + elop) + elop).to_matrix(dims),
                       [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]])
    assert np.allclose((elop + (scop + elop)).to_matrix(dims),
                       [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]])
    assert np.allclose(
        ((scop - elop) - scop).to_matrix(dims),
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    assert np.allclose((scop - (scop - elop)).to_matrix(dims),
                       [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.allclose(((scop - elop) - elop).to_matrix(dims),
                       [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert np.allclose((elop - (scop - elop)).to_matrix(dims),
                       [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    assert np.allclose(operator.add(scop, 2).to_matrix(dims), [4. + 0.j])
    assert np.allclose(operator.add(scop, 2.5).to_matrix(dims), [4.5 + 0j])
    assert np.allclose(operator.add(scop, 2j).to_matrix(dims), [2 + 2j])
    assert np.allclose(operator.add(2, scop).to_matrix(dims), [4 + 0j])
    assert np.allclose(operator.add(2.5, scop).to_matrix(dims), [4.5 + 0j])
    assert np.allclose(operator.add(2j, scop).to_matrix(dims), [2 + 2j])

    assert np.allclose(operator.sub(scop, 2).to_matrix(dims), [0 + 0.j])
    assert np.allclose(operator.sub(scop, 2.5).to_matrix(dims), [-0.5 + 0j])
    assert np.allclose(operator.sub(scop, 2j).to_matrix(dims), [2 - 2j])
    assert np.allclose(operator.sub(2, scop).to_matrix(dims), [0 + 0j])
    assert np.allclose(operator.sub(2.5, scop).to_matrix(dims), [0.5 + 0j])
    assert np.allclose(operator.sub(2j, scop).to_matrix(dims), [-2 + 2j])

    assert np.allclose(operator.mul(scop, 2).to_matrix(dims), [4 + 0.j])
    assert np.allclose(operator.mul(scop, 2.5).to_matrix(dims), [5 + 0j])
    assert np.allclose(operator.mul(scop, 2j).to_matrix(dims), [0 + 4j])
    assert np.allclose(operator.mul(2, scop).to_matrix(dims), [4 + 0j])
    assert np.allclose(operator.mul(2.5, scop).to_matrix(dims), [5 + 0j])
    assert np.allclose(operator.mul(2j, scop).to_matrix(dims), [0 + 4j])

    assert np.allclose(operator.truediv(scop, 2).to_matrix(dims), [1 + 0.j])
    assert np.allclose(operator.truediv(scop, 2.5).to_matrix(dims), [0.8 + 0j])
    assert np.allclose(operator.truediv(scop, 2j).to_matrix(dims), [0 - 1j])
    assert np.allclose(operator.truediv(2, scop).to_matrix(dims), [1 + 0j])
    assert np.allclose(operator.truediv(2.5, scop).to_matrix(dims), [1.25 + 0j])
    assert np.allclose(operator.truediv(2j, scop).to_matrix(dims), [0 + 1j])

    assert np.allclose(operator.pow(scop, 2).to_matrix(dims), [4 + 0j])
    assert np.allclose(
        operator.pow(scop, 2.5).to_matrix(dims), [5.65685425 + 0j])
    assert np.allclose(
        operator.pow(scop, 2j).to_matrix(dims), [0.18345697 + 0.98302774j])
    assert np.allclose(operator.pow(2, scop).to_matrix(dims), [4 + 0j])
    assert np.allclose(operator.pow(2.5, scop).to_matrix(dims), [6.25 + 0j])
    assert np.allclose(operator.pow(2j, scop).to_matrix(dims), [-4 + 0j])

    opprod = operators.create(0) * operators.annihilate(0)
    opsum = operators.create(0) + operators.annihilate(0)

    assert np.allclose(
        operator.add(elop, 2).to_matrix(dims),
        [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]])
    assert np.allclose(
        operator.add(elop, 2.5).to_matrix(dims),
        [[3.5, 0, 0, 0], [0, 3.5, 0, 0], [0, 0, 3.5, 0], [0, 0, 0, 3.5]])
    assert np.allclose(
        operator.add(elop, 2j).to_matrix(dims),
        [[1 + 2j, 0, 0, 0], [0, 1 + 2j, 0, 0], [0, 0, 1 + 2j, 0],
         [0, 0, 0, 1 + 2j]])
    assert np.allclose(
        operator.add(2, elop).to_matrix(dims),
        [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]])
    assert np.allclose(
        operator.add(2.5, elop).to_matrix(dims),
        [[3.5, 0, 0, 0], [0, 3.5, 0, 0], [0, 0, 3.5, 0], [0, 0, 0, 3.5]])
    assert np.allclose(
        operator.add(2j, elop).to_matrix(dims),
        [[1 + 2j, 0, 0, 0], [0, 1 + 2j, 0, 0], [0, 0, 1 + 2j, 0],
         [0, 0, 0, 1 + 2j]])

    assert np.allclose(
        operator.sub(elop, 2).to_matrix(dims),
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    assert np.allclose(
        operator.sub(elop, 2.5).to_matrix(dims),
        [[-1.5, 0, 0, 0], [0, -1.5, 0, 0], [0, 0, -1.5, 0], [0, 0, 0, -1.5]])
    assert np.allclose(
        operator.sub(elop, 2j).to_matrix(dims),
        [[1 - 2j, 0, 0, 0], [0, 1 - 2j, 0, 0], [0, 0, 1 - 2j, 0],
         [0, 0, 0, 1 - 2j]])
    assert np.allclose(
        operator.sub(2, elop).to_matrix(dims),
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.allclose(
        operator.sub(2.5, elop).to_matrix(dims),
        [[1.5, 0, 0, 0], [0, 1.5, 0, 0], [0, 0, 1.5, 0], [0, 0, 0, 1.5]])
    assert np.allclose(
        operator.sub(2j, elop).to_matrix(dims),
        [[-1 + 2j, 0, 0, 0], [0, -1 + 2j, 0, 0], [0, 0, -1 + 2j, 0],
         [0, 0, 0, -1 + 2j]])

    assert np.allclose(
        operator.mul(elop, 2).to_matrix(dims),
        [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    assert np.allclose(
        operator.mul(elop, 2.5).to_matrix(dims),
        [[2.5, 0, 0, 0], [0, 2.5, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 2.5]])
    assert np.allclose(
        operator.mul(elop, 2j).to_matrix(dims),
        [[2j, 0, 0, 0], [0, 2j, 0, 0], [0, 0, 2j, 0], [0, 0, 0, 2j]])
    assert np.allclose(
        operator.mul(2, elop).to_matrix(dims),
        [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    assert np.allclose(
        operator.mul(2.5, elop).to_matrix(dims),
        [[2.5, 0, 0, 0], [0, 2.5, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 2.5]])
    assert np.allclose(
        operator.mul(2j, elop).to_matrix(dims),
        [[2j, 0, 0, 0], [0, 2j, 0, 0], [0, 0, 2j, 0], [0, 0, 0, 2j]])

    assert np.allclose(
        (elop / 2).to_matrix(dims),
        [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]])
    assert np.allclose(
        (elop / 2.5).to_matrix(dims),
        [[0.4, 0, 0, 0], [0, 0.4, 0, 0], [0, 0, 0.4, 0], [0, 0, 0, 0.4]])
    assert np.allclose((elop / 2j).to_matrix(dims),
                       [[-0.5j, 0, 0, 0], [0, -0.5j, 0, 0], [0, 0, -0.5j, 0],
                        [0, 0, 0, -0.5j]])

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

    assert np.allclose(operator.add(opsum, 2).to_matrix(dims), [[2, 1], [1, 2]])
    assert np.allclose(
        operator.add(opsum, 2.5).to_matrix(dims), [[2.5, 1], [1, 2.5]])
    assert np.allclose(
        operator.add(opsum, 2j).to_matrix(dims), [[2j, 1], [1, 2j]])
    assert np.allclose(operator.add(2, opsum).to_matrix(dims), [[2, 1], [1, 2]])
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

    assert np.allclose(operator.mul(opsum, 2).to_matrix(dims), [[0, 2], [2, 0]])
    assert np.allclose(
        operator.mul(opsum, 2.5).to_matrix(dims), [[0, 2.5], [2.5, 0]])
    assert np.allclose(
        operator.mul(opsum, 2j).to_matrix(dims), [[0, 2j], [2j, 0]])
    assert np.allclose(operator.mul(2, opsum).to_matrix(dims), [[0, 2], [2, 0]])
    assert np.allclose(
        operator.mul(2.5, opsum).to_matrix(dims), [[0, 2.5], [2.5, 0]])
    assert np.allclose(
        operator.mul(2j, opsum).to_matrix(dims), [[0, 2j], [2j, 0]])

    assert np.allclose((opsum / 2).to_matrix(dims), [[0, 0.5], [0.5, 0]])
    assert np.allclose((opsum / 2.5).to_matrix(dims), [[0, 0.4], [0.4, 0]])
    assert np.allclose((opsum / 2j).to_matrix(dims), [[0, -0.5j], [-0.5j, 0]])


def test_elementary_operator():
    dims = {0: 2, 1: 2}
    elop = ElementaryOperator.identity(1)
    scop = ScalarOperator.const(2)
    assert np.allclose((elop + scop).to_matrix(dims),
                       (scop + elop).to_matrix(dims))
    assert np.allclose((elop * scop).to_matrix(dims),
                       (scop * elop).to_matrix(dims))


def test_transverse_field():
    dims = {0: 2, 1: 2, 2: 2}
    num_qubits = 3
    field_strength = ScalarOperator(lambda t: t)
    transverse_op = sum(field_strength * spin.x(i) for i in range(num_qubits))
    assert transverse_op.to_matrix(dims, t=1.0).shape == (8, 8)


def test_ising_chain():
    dims = {0: 2, 1: 2, 2: 2}
    num_qubits = 3
    coupling_strength = ScalarOperator(lambda t: t)
    ising_op = sum(coupling_strength * spin.z(i) * spin.z(i + 1)
                   for i in range(num_qubits - 1))
    assert ising_op.to_matrix(dims, t=1.0).shape == (8, 8)


# Run with: pytest -rP
if __name__ == "__main__":
    pytest.main(["-rP"])
