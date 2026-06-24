# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np, operator, pytest
from cudaq.operators import ScalarOperator
from cudaq import spin, operators


def test_construction():
    const = operators.const(3)
    assert const.is_constant()
    assert const.evaluate() == 3
    assert np.allclose(const.to_matrix(), [3])
    assert str(const) == '(3+0i)'

    const = operators.const(3.)
    assert const.is_constant()
    assert const.evaluate() == 3
    assert np.allclose(const.to_matrix(), [3])
    assert str(const) == '(3+0i)'

    const = operators.const(3j)
    assert const.is_constant()
    assert const.evaluate() == 3j
    assert np.allclose(const.to_matrix(), [3j])
    assert str(const) == '(0+3i)'

    const = operators.const(np.complex128(3))
    assert const.is_constant()
    assert const.evaluate() == 3
    assert np.allclose(const.to_matrix(), [3])
    assert str(const) == '(3+0i)'

    const = ScalarOperator.const(5)
    assert const.is_constant()
    assert const.evaluate() == 5
    assert np.allclose(const.to_matrix(), [5])
    assert str(const) == '(5+0i)'

    def callback(x):
        return x * x

    fct = ScalarOperator(callback)
    assert not fct.is_constant()
    with pytest.raises(Exception):
        fct.evaluate()
    with pytest.raises(Exception):
        fct.evaluate(y=3)
    assert fct.evaluate(x=3) == 9
    assert fct.evaluate(x=5) == 25
    with pytest.raises(Exception):
        fct.to_matrix()
    assert np.allclose(fct.to_matrix(x=3), [9])
    assert np.allclose(fct.to_matrix(x=5), [25])
    assert str(fct) == 'scalar(x)'
    assert 'x' in fct.parameters

    lam = ScalarOperator(lambda z: z - 1)
    assert not lam.is_constant()
    with pytest.raises(Exception):
        lam.evaluate()
    with pytest.raises(Exception):
        lam.evaluate(y=3)
    assert lam.evaluate(z=3) == 2
    assert lam.evaluate(z=5) == 4
    with pytest.raises(Exception):
        lam.to_matrix()
    assert np.allclose(lam.to_matrix(z=3), [2])
    assert np.allclose(lam.to_matrix(z=5), [4])
    assert str(lam) == 'scalar(z)'
    assert 'z' in lam.parameters


def test_composition():

    so1 = ScalarOperator(lambda t: t)
    so2 = ScalarOperator(lambda t: t - 1)
    so3 = ScalarOperator(lambda p: 2 * p)

    sp1 = so1 * so2
    sp2 = so2 * so3

    with pytest.raises(Exception):
        sp1.evaluate()
    assert sp1.evaluate(t=3) == 6
    assert len(sp1.parameters) == 1
    assert 't' in sp1.parameters

    with pytest.raises(Exception):
        sp2.evaluate(t=3)
    assert sp2.evaluate(t=3, p=4) == 16
    assert sp2.evaluate(p=4, t=3) == 16
    assert len(sp2.parameters) == 2
    assert 't' in sp2.parameters
    assert 'p' in sp2.parameters

    sd1 = so1 / so2
    sd2 = so2 / so3

    with pytest.raises(Exception):
        sd1.evaluate()
    assert sd1.evaluate(t=3) == 1.5
    assert len(sd1.parameters) == 1
    assert 't' in sd1.parameters

    with pytest.raises(Exception):
        sd2.evaluate(t=3)
    assert sd2.evaluate(t=3, p=4) == 0.25
    assert sd2.evaluate(p=4, t=3) == 0.25
    assert len(sd2.parameters) == 2
    assert 't' in sd2.parameters
    assert 'p' in sd2.parameters

    sa1 = so1 + so2
    sa2 = so2 + so3

    with pytest.raises(Exception):
        sa1.evaluate()
    assert sa1.evaluate(t=3) == 5
    assert len(sa1.parameters) == 1
    assert 't' in sa1.parameters

    with pytest.raises(Exception):
        sa2.evaluate(t=3)
    assert sa2.evaluate(t=3, p=4) == 10
    assert sa2.evaluate(p=4, t=3) == 10
    assert len(sa2.parameters) == 2
    assert 't' in sa2.parameters
    assert 'p' in sa2.parameters

    ss1 = so1 - so2
    ss2 = so2 - so3

    with pytest.raises(Exception):
        ss1.evaluate()
    assert ss1.evaluate(t=3) == 1
    assert len(ss1.parameters) == 1
    assert 't' in ss1.parameters

    with pytest.raises(Exception):
        ss2.evaluate(t=3)
    assert ss2.evaluate(t=3, p=4) == -6
    assert ss2.evaluate(p=4, t=3) == -6
    assert len(ss2.parameters) == 2
    assert 't' in ss2.parameters
    assert 'p' in ss2.parameters

    se1 = so1**so2
    se2 = so3**so2

    with pytest.raises(Exception):
        se1.evaluate()
    assert se1.evaluate(t=3) == 9
    assert len(se1.parameters) == 1
    assert 't' in se1.parameters

    with pytest.raises(Exception):
        se2.evaluate(t=3)
    assert se2.evaluate(t=3, p=4) == 64
    assert se2.evaluate(p=4, t=3) == 64
    assert len(se2.parameters) == 2
    assert 't' in se2.parameters
    assert 'p' in se2.parameters


def test_parameter_docs():

    def generator(t):
        """Time dependent operator where t is the time"""
        return t

    so = ScalarOperator(generator)
    assert 't' in so.parameters
    assert so.parameters['t'] == ""  # no docs for t

    def generator1(arg):
        """Scalar function with an arg parameter.
        Args:

        arg: Description of `arg`. Multiple
                lines are supported.
        Returns:
        Something that depends on arg.
        """
        return arg

    def generator2(params):
        """
        Arguments:
        params: Description of params
        """
        return params[0] * 2

    so1 = ScalarOperator(generator1)
    so2 = ScalarOperator(generator2)
    sp = so1 * so2

    assert len(sp.parameters) == 2
    assert 'arg' in sp.parameters
    assert sp.parameters[
        'arg'] == "Description of `arg`. Multiple lines are supported."
    assert 'params' in sp.parameters
    assert sp.parameters['params'] == "Description of params"
    sp.parameters['arg'] = "New docs"
    assert so1.parameters[
        'arg'] == "Description of `arg`. Multiple lines are supported."
    assert sp.parameters[
        'arg'] == "Description of `arg`. Multiple lines are supported."

    # multiple parameters with docs
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

    op3 = ScalarOperator(generator)
    assert 'param1' in op3.parameters
    assert op3.parameters['param1'] == 'my docs for param1'
    assert 'args' in op3.parameters
    assert op3.parameters[
        'args'] == 'Description of `args`. Multiple lines are supported.'


def test_equality():

    assert ScalarOperator.const(5) == ScalarOperator.const(5)
    assert ScalarOperator.const(5) == ScalarOperator.const(5 + 0j)
    assert ScalarOperator.const(5) != ScalarOperator.const(5j)
    assert ScalarOperator(lambda: 5) != ScalarOperator.const(5)
    assert ScalarOperator(lambda: 5) != ScalarOperator(lambda: 5)

    # Note that while it would be preferable if we could identity
    # when the same generator is used, the bindings construct a new
    # function such that this is not currently recognized.
    generator = lambda: 5
    assert ScalarOperator(generator) != ScalarOperator(lambda: 5)
    assert ScalarOperator(generator) != ScalarOperator(generator)
    # The same problem also applies as soon as a non-constant scalar
    # is multiplied/added/etc. with another operator.
    so = ScalarOperator(generator)
    assert so == so
    assert spin.x(0) * so != spin.x(0) * so
    assert spin.x(0) * so != so * spin.x(0)

    assert (ScalarOperator.const(5) +
            ScalarOperator.const(3)) == (ScalarOperator.const(4) +
                                         ScalarOperator.const(4))
    assert (ScalarOperator.const(6) *
            ScalarOperator.const(2)) == (ScalarOperator.const(4) *
                                         ScalarOperator.const(3))

    elop = spin.x(0)
    assert ((ScalarOperator.const(5) + ScalarOperator.const(3)) *
            elop) == (elop *
                      (ScalarOperator.const(4) + ScalarOperator.const(4)))
    assert (ScalarOperator.const(6) * ScalarOperator.const(2) +
            elop) == (elop + ScalarOperator.const(4) * ScalarOperator.const(3))
    assert (ScalarOperator.const(6) * ScalarOperator.const(2) *
            elop) == (elop * ScalarOperator.const(4) * ScalarOperator.const(3))
    assert (ScalarOperator.const(5) + 3) == (4 + ScalarOperator.const(4))
    assert (ScalarOperator.const(6) * 2) == (4 * ScalarOperator.const(3))
    assert ((ScalarOperator.const(5) + 3) *
            elop) == (elop * (4 + ScalarOperator.const(4)))
    assert (ScalarOperator.const(6) * 2 + elop) == (elop +
                                                    4 * ScalarOperator.const(3))
    assert (ScalarOperator.const(6) * 2.0 * elop) == (elop * 4.0 *
                                                      ScalarOperator.const(3))
    assert (ScalarOperator.const(6) / 2) == ScalarOperator.const(3)


def test_arithmetics():
    so1 = ScalarOperator.const(3)
    so2 = ScalarOperator.const(2)

    assert np.allclose((so1 + so2).to_matrix(), [5])
    assert np.allclose((so1 - so2).to_matrix(), [1])
    assert np.allclose((so1 * so2).to_matrix(), [6])
    assert np.allclose((so1 / so2).to_matrix(), [1.5])
    assert np.allclose((so1**2).to_matrix(), [9])

    scop = ScalarOperator.const(2)
    assert np.allclose(operator.add(scop, 2).to_matrix(), [4. + 0.j])
    assert np.allclose(operator.add(scop, 2.5).to_matrix(), [4.5 + 0j])
    assert np.allclose(operator.add(scop, 2j).to_matrix(), [2 + 2j])
    assert np.allclose(operator.add(2, scop).to_matrix(), [4 + 0j])
    assert np.allclose(operator.add(2.5, scop).to_matrix(), [4.5 + 0j])
    assert np.allclose(operator.add(2j, scop).to_matrix(), [2 + 2j])

    assert np.allclose(operator.sub(scop, 2).to_matrix(), [0 + 0.j])
    assert np.allclose(operator.sub(scop, 2.5).to_matrix(), [-0.5 + 0j])
    assert np.allclose(operator.sub(scop, 2j).to_matrix(), [2 - 2j])
    assert np.allclose(operator.sub(2, scop).to_matrix(), [0 + 0j])
    assert np.allclose(operator.sub(2.5, scop).to_matrix(), [0.5 + 0j])
    assert np.allclose(operator.sub(2j, scop).to_matrix(), [-2 + 2j])

    assert np.allclose(operator.mul(scop, 2).to_matrix(), [4 + 0.j])
    assert np.allclose(operator.mul(scop, 2.5).to_matrix(), [5 + 0j])
    assert np.allclose(operator.mul(scop, 2j).to_matrix(), [0 + 4j])
    assert np.allclose(operator.mul(2, scop).to_matrix(), [4 + 0j])
    assert np.allclose(operator.mul(2.5, scop).to_matrix(), [5 + 0j])
    assert np.allclose(operator.mul(2j, scop).to_matrix(), [0 + 4j])

    assert np.allclose(operator.truediv(scop, 2).to_matrix(), [1 + 0.j])
    assert np.allclose(operator.truediv(scop, 2.5).to_matrix(), [0.8 + 0j])
    assert np.allclose(operator.truediv(scop, 2j).to_matrix(), [0 - 1j])
    assert np.allclose(operator.truediv(2, scop).to_matrix(), [1 + 0j])
    assert np.allclose(operator.truediv(2.5, scop).to_matrix(), [1.25 + 0j])
    assert np.allclose(operator.truediv(2j, scop).to_matrix(), [0 + 1j])

    assert np.allclose(operator.pow(scop, 2).to_matrix(), [4 + 0j])
    assert np.allclose(operator.pow(scop, 2.5).to_matrix(), [5.65685425 + 0j])
    assert np.allclose(
        operator.pow(scop, 2j).to_matrix(), [0.18345697 + 0.98302774j])
    assert np.allclose(operator.pow(2, scop).to_matrix(), [4 + 0j])
    assert np.allclose(operator.pow(2.5, scop).to_matrix(), [6.25 + 0j])
    assert np.allclose(operator.pow(2j, scop).to_matrix(), [-4 + 0j])

    so1 = ScalarOperator(lambda t: t)
    assert so1.to_matrix(t=2.0) == 2.0
    op1 = so1 * spin.x(0)
    op2 = spin.x(0) * so1
    op3 = so1 + spin.x(0)
    op4 = spin.x(0) + so1
    assert np.allclose(op1.to_matrix(t=2.0), [[0, 2], [2, 0]])
    assert np.allclose(op2.to_matrix(t=2.0), [[0, 2], [2, 0]])
    assert np.allclose(op3.to_matrix(t=2.0), [[2, 1], [1, 2]])
    assert np.allclose(op4.to_matrix(t=2.0), [[2, 1], [1, 2]])
    assert np.allclose(op1.to_matrix(t=1j), [[0, 1j], [1j, 0]])
    assert np.allclose(op2.to_matrix(t=1j), [[0, 1j], [1j, 0]])
    assert np.allclose(op3.to_matrix(t=1j), [[1j, 1], [1, 1j]])
    assert np.allclose(op4.to_matrix(t=1j), [[1j, 1], [1, 1j]])

    dims = {0: 2}
    assert so1.to_matrix(dims, t=2.0) == 2.0
    assert np.allclose(op1.to_matrix(dims, t=2.0), [[0, 2], [2, 0]])
    assert np.allclose(op2.to_matrix(dims, t=2.0), [[0, 2], [2, 0]])
    assert np.allclose(op3.to_matrix(dims, t=2.0), [[2, 1], [1, 2]])
    assert np.allclose(op4.to_matrix(dims, t=2.0), [[2, 1], [1, 2]])


# for debugging
if __name__ == "__main__":
    test_parameter_docs()
    #pytest.main(["-rP"])
