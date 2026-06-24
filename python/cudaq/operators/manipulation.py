# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generator, Generic, Iterable, Mapping, TypeVar

from .definitions import NumericType, OperatorSum, ProductOperator, ElementaryOperator
from .scalar import ScalarOperator

TEval = TypeVar('TEval')


class OperatorArithmetics(ABC, Generic[TEval]):
    """
    This class serves as a monad base class for computing arbitrary values
    during operator evaluation.
    """

    @abstractmethod
    def evaluate(self: OperatorArithmetics[TEval],
                 op: ElementaryOperator | ScalarOperator) -> TEval:
        """
        Accesses the relevant data to evaluate an operator expression in the leaf 
        nodes, that is in elementary and scalar operators.
        """
        pass

    @abstractmethod
    def add(self: OperatorArithmetics[TEval], val1: TEval,
            val2: TEval) -> TEval:
        """
        Adds two operators that act on the same degrees of freedom.
        """
        pass

    @abstractmethod
    def mul(self: OperatorArithmetics[TEval], val1: TEval,
            val2: TEval) -> TEval:
        """
        Multiplies two operators that act on the same degrees of freedom.
        """
        pass

    @abstractmethod
    def tensor(self: OperatorArithmetics[TEval], val1: TEval,
               val2: TEval) -> TEval:
        """
        Computes the tensor product of two operators that act on different 
        degrees of freedom.
        """
        pass


class OperatorEvaluation(OperatorArithmetics[OperatorSum | ProductOperator |
                                             NumericType]):

    def tensor(
        self: OperatorEvaluation,
        op1: OperatorSum | ProductOperator | NumericType,
        op2: OperatorSum | ProductOperator | NumericType
    ) -> OperatorSum | ProductOperator | NumericType:
        return op1 * op2

    def mul(
        self: OperatorEvaluation,
        op1: OperatorSum | ProductOperator | NumericType,
        op2: OperatorSum | ProductOperator | NumericType
    ) -> OperatorSum | ProductOperator | NumericType:
        return op1 * op2

    def add(
        self: OperatorEvaluation,
        op1: OperatorSum | ProductOperator | NumericType,
        op2: OperatorSum | ProductOperator | NumericType
    ) -> OperatorSum | ProductOperator | NumericType:
        return op1 + op2

    def evaluate(
        self: OperatorEvaluation, op: ElementaryOperator | ScalarOperator
    ) -> ProductOperator | NumericType:
        if isinstance(op, ElementaryOperator):
            if hasattr(op, "evaluate"):
                return self._term_type(op.evaluate(**self._kwargs))
            else:
                return self._term_type(op)
        if isinstance(op, ScalarOperator):
            return op.evaluate(**self._kwargs)
        else:
            raise ValueError(
                f"operator '{str(op)}' (type: {type(op)}) is not a scalar or elementary operator"
            )

    def __init__(self: OperatorEvaluation, term_type,
                 **kwargs: NumericType) -> None:
        """
        Instantiates a SpinEvaluation instance for the given keyword arguments.
        This class is only defined for qubits, that is all degrees of freedom must 
        have dimension two.

        Arguments:
            `kwargs`: Keyword arguments needed to evaluate, that is access data in,
                the leaf nodes of the operator expression. Leaf nodes are 
                elementary or scalar operators.
        """
        self._kwargs = kwargs
        self._term_type = term_type


def _product_transformation(term: ProductOperator,
                            arithmetics: OperatorArithmetics[TEval],
                            pad_terms: bool = True):
    """
    Helper function used for evaluating operator expressions and computing arbitrary values
    during evaluation. The value to be computed is defined by the OperatorArithmetics.
    The evaluation guarantees that addition and multiplication of two operators will only
    be called when both operators act on the same degrees of freedom, and the tensor product
    will only be computed if they act on different degrees of freedom. 
    """

    def padded_op(op: ElementaryOperator | ScalarOperator,
                  degrees: Iterable[int]):
        # Creating the tensor product with op being last is most efficient.
        def accumulate_ops() -> Generator[TEval]:
            op_degrees = op.degrees
            for degree in degrees:
                if not degree in op_degrees:
                    yield arithmetics.evaluate(op.__class__(degree))
            yield arithmetics.evaluate(op)

        evaluated_ops = accumulate_ops()
        padded = next(evaluated_ops)
        for value in evaluated_ops:
            padded = arithmetics.tensor(padded, value)
        return padded

    evaluated = arithmetics.evaluate(term.coefficient)
    if pad_terms:
        degrees = term.degrees
        for op in term:
            if op != op.__class__(op.degrees[0]):
                evaluated = arithmetics.mul(evaluated, padded_op(op, degrees))
    else:
        for op in term:
            evaluated = arithmetics.mul(evaluated, arithmetics.evaluate(op))
    return evaluated


def _sum_transformation(operator: OperatorSum,
                        arithmetics: OperatorArithmetics[TEval],
                        pad_terms: bool = True):
    """
    Helper function used for evaluating operator expressions and computing arbitrary values
    during evaluation. The value to be computed is defined by the OperatorArithmetics.
    The evaluation guarantees that addition and multiplication of two operators will only
    be called when both operators act on the same degrees of freedom, and the tensor product
    will only be computed if they act on different degrees of freedom. 
    """

    identity = operator.__class__.identity

    def padded_term(term: ProductOperator,
                    degrees: Iterable[int]) -> ProductOperator:
        term_degrees = term.degrees
        padded_term = term.copy()
        for degree in degrees:
            if degree not in term_degrees:
                padded_term *= identity(degree)
        return padded_term

    evaluated = arithmetics.evaluate(ScalarOperator.const(0))
    if pad_terms:
        degrees = operator.degrees
        for term in operator:
            evaluated_term = _product_transformation(padded_term(term, degrees),
                                                     arithmetics, pad_terms)
            evaluated = arithmetics.add(evaluated, evaluated_term)
    else:
        for term in operator:
            evaluated_term = _product_transformation(term, arithmetics,
                                                     pad_terms)
            evaluated = arithmetics.add(evaluated, evaluated_term)
    return evaluated


def _evaluate(operator: OperatorSum | ProductOperator,
              **kwargs: NumericType) -> OperatorSum | ProductOperator:
    term_type = type(operator)
    if isinstance(operator, OperatorSum) and operator.term_count > 0:
        term, *_ = operator
        term_type = type(term)
    arithmetics = OperatorEvaluation(term_type, **kwargs)
    if isinstance(operator, OperatorSum):
        evaluated = _sum_transformation(operator, arithmetics, False)
    elif isinstance(operator, ProductOperator):
        evaluated = _product_transformation(operator, arithmetics, False)
    else:
        raise RuntimeError("the given value is not an operator")

    if isinstance(evaluated, OperatorSum) or isinstance(evaluated,
                                                        ProductOperator):
        return evaluated
    elif isinstance(operator, OperatorSum):
        evaluated_sum = operator.__class__.empty()
        if evaluated != 0:
            evaluated_sum += evaluated
        return evaluated_sum
    else:
        return operator.__class__(evaluated)
