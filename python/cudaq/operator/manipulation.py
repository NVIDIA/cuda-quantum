# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
import numpy, re, sys  # type: ignore
from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar, Tuple
from numpy.typing import NDArray
from functools import lru_cache

from .helpers import _OperatorHelpers, NumericType
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime

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


class MatrixArithmetics(OperatorArithmetics['MatrixArithmetics.Evaluated']):
    """
    Encapsulates the functions needed to compute the matrix representation
    of an operator expression.
    """

    class Evaluated:
        """
        Stores the relevant data to compute the matrix representation of an
        operator expression.
        """

        def __init__(self: MatrixArithmetics.Evaluated, degrees: Iterable[int],
                     matrix: NDArray[numpy.complexfloating]) -> None:
            """
            Instantiates an object that contains the matrix representation of an
            operator acting on the given degrees of freedom.

            Arguments:
                degrees: The degrees of freedom that the matrix applies to.
                matrix: The matrix representation of an evaluated operator.
            """
            self._degrees = tuple(degrees)
            self._matrix = matrix

        @property
        def degrees(self: MatrixArithmetics.Evaluated) -> Tuple[int]:
            """
            The degrees of freedom that the matrix of the evaluated value applies to.
            """
            return self._degrees

        @property
        def matrix(
            self: MatrixArithmetics.Evaluated
        ) -> NDArray[numpy.complexfloating]:
            """
            The matrix representation of an evaluated operator, ordered according
            to the sequence of degrees of freedom associated with the evaluated value.
            """
            return self._matrix

    @lru_cache(maxsize=None)
    def _compute_permutation(self, op_degrees, canon_degrees):
        states = _OperatorHelpers.generate_all_states(canon_degrees,
                                                      self._dimensions)
        reordering = [canon_degrees.index(deg) for deg in op_degrees]
        return [
            states.index(''.join(state[i]
                                 for i in reordering))
            for state in states
        ]

    def _canonicalize(
        self: MatrixArithmetics, op_matrix: NDArray[numpy.complexfloating],
        op_degrees: Iterable[int]
    ) -> Tuple[NDArray[numpy.complexfloating], Tuple[int]]:
        """
        Given a matrix representation that acts on the given degrees or freedom, 
        sorts the degrees and permutes the matrix to match that canonical order.

        Returns:
            A tuple consisting of the permuted matrix as well as the sequence of degrees
            of freedom in canonical order.
        """
        canon_degrees = _OperatorHelpers.canonicalize_degrees(op_degrees)
        if op_degrees == canon_degrees:
            return op_matrix, canon_degrees

        permutation = self._compute_permutation(tuple(op_degrees),
                                                tuple(canon_degrees))
        # [states[i] for i in permutation] produces op_states
        _OperatorHelpers.permute_matrix(op_matrix, permutation)
        return op_matrix, canon_degrees

    def tensor(self: MatrixArithmetics, op1: MatrixArithmetics.Evaluated,
               op2: MatrixArithmetics.Evaluated) -> MatrixArithmetics.Evaluated:
        """
        Computes the tensor product of two evaluate operators that act on different 
        degrees of freedom using `numpy.kron`.
        """
        assert len(frozenset(op1.degrees).intersection(op2.degrees)) == 0, \
            "Operators should not have common degrees of freedom."
        op_degrees = op1.degrees + op2.degrees
        op_matrix = numpy.kron(op1.matrix, op2.matrix)
        new_matrix, new_degrees = self._canonicalize(op_matrix, op_degrees)
        return MatrixArithmetics.Evaluated(new_degrees, new_matrix)

    def mul(self: MatrixArithmetics, op1: MatrixArithmetics.Evaluated,
            op2: MatrixArithmetics.Evaluated) -> MatrixArithmetics.Evaluated:
        """
        Multiplies two evaluated operators that act on the same degrees of freedom
        using `numpy.dot`.
        """
        # Elementary operators have sorted degrees such that we have a unique convention
        # for how to define the matrix. Tensor products permute the computed matrix if
        # necessary to guarantee that all operators always have sorted degrees.
        assert op1.degrees == op2.degrees, "Operators should have the same order of degrees."
        return MatrixArithmetics.Evaluated(op1.degrees,
                                           numpy.dot(op1.matrix, op2.matrix))

    def add(self: MatrixArithmetics, op1: MatrixArithmetics.Evaluated,
            op2: MatrixArithmetics.Evaluated) -> MatrixArithmetics.Evaluated:
        """
        Multiplies two evaluated operators that act on the same degrees of freedom
        using `numpy`'s array addition.
        """
        # Elementary operators have sorted degrees such that we have a unique convention
        # for how to define the matrix. Tensor products permute the computed matrix if
        # necessary to guarantee that all operators always have sorted degrees.
        assert op1.degrees == op2.degrees, "Operators should have the same order of degrees."
        return MatrixArithmetics.Evaluated(op1.degrees, op1.matrix + op2.matrix)

    def evaluate(
            self: MatrixArithmetics, op: ElementaryOperator | ScalarOperator
    ) -> MatrixArithmetics.Evaluated:
        """
        Computes the matrix of an ElementaryOperator or ScalarOperator using its 
        `to_matrix` method.
        """
        matrix = op.to_matrix(self._dimensions, **self._kwargs)
        return MatrixArithmetics.Evaluated(op._degrees, matrix)

    def __init__(self: MatrixArithmetics, dimensions: Mapping[int, int],
                 **kwargs: NumericType) -> None:
        """
        Instantiates a MatrixArithmetics instance that can act on the given
        dimensions.

        Arguments:
            dimensions: A mapping that specifies the number of levels, that 
                is the dimension, of each degree of freedom that the evaluated 
                operator can act on.
            `kwargs`: Keyword arguments needed to evaluate, that is access data in,
                the leaf nodes of the operator expression. Leaf nodes are 
                values of type ElementaryOperator or ScalarOperator.
        """
        self._dimensions = dimensions
        self._kwargs = kwargs


class PrettyPrint(OperatorArithmetics[str]):

    def tensor(self, op1: str, op2: str) -> str:

        def add_parens(str_value: str):
            if any(op in str_value for op in (" + ", " * ")):
                return f"({str_value})"
            return str_value

        return f"{add_parens(op1)} x {add_parens(op2)}"

    def mul(self, op1: str, op2: str) -> str:

        def add_parens(str_value: str):
            outer_str = re.sub(r'\(.+?\)', '', str_value)
            if " + " in outer_str or " x " in outer_str:
                return f"({str_value})"
            else:
                return str_value

        return f"{add_parens(op1)} * {add_parens(op2)}"

    def add(self, op1: str, op2: str) -> str:
        return f"{op1} + {op2}"

    def evaluate(self, op: ElementaryOperator | ScalarOperator) -> str:
        return str(op)


# FIXME(OperatorCpp): To be removed/replaced. We need to be able to pass general operators to cudaq.observe.
class _SpinArithmetics(OperatorArithmetics[cudaq_runtime.SpinOperator |
                                           NumericType]):

    def tensor(
        self: _SpinArithmetics, op1: cudaq_runtime.SpinOperator | NumericType,
        op2: cudaq_runtime.SpinOperator | NumericType
    ) -> cudaq_runtime.SpinOperator | NumericType:
        return op1 * op2

    def mul(
        self: _SpinArithmetics, op1: cudaq_runtime.SpinOperator | NumericType,
        op2: cudaq_runtime.SpinOperator | NumericType
    ) -> cudaq_runtime.SpinOperator | NumericType:
        return op1 * op2

    def add(
        self: _SpinArithmetics, op1: cudaq_runtime.SpinOperator | NumericType,
        op2: cudaq_runtime.SpinOperator | NumericType
    ) -> cudaq_runtime.SpinOperator | NumericType:
        # FIXME(OperatorCpp): `SpinOperator` only exposes `+` operator for `double`, needs to multiply with an identity operator before adding.
        if isinstance(op1, NumericType) and isinstance(
                op2, cudaq_runtime.SpinOperator):
            return op1 * cudaq_runtime.SpinOperator() + op2
        if isinstance(op2, NumericType) and isinstance(
                op1, cudaq_runtime.SpinOperator):
            return op2 * cudaq_runtime.SpinOperator() + op1
        return op1 + op2

    def evaluate(
        self: _SpinArithmetics, op: ElementaryOperator | ScalarOperator
    ) -> cudaq_runtime.SpinOperator | NumericType:
        op_id = getattr(op, "id", "scalar")
        if op_id == "scalar":
            return op.evaluate(**self._kwargs)
        elif op_id == "pauli_x":
            return cudaq_runtime.spin.x(op.degrees[0])
        elif op_id == "pauli_y":
            return cudaq_runtime.spin.y(op.degrees[0])
        elif op_id == "pauli_z":
            return cudaq_runtime.spin.z(op.degrees[0])
        elif op_id == "pauli_i":
            return cudaq_runtime.spin.i(op.degrees[0])
        elif op_id == "identity":
            assert len(
                op.degrees
            ) == 1, "expecting identity to act on a single degree of freedom"
            return cudaq_runtime.spin.i(op.degrees[0])
        else:
            raise ValueError(f"operator '{op_id}' is not a spin operator")

    def __init__(self: _SpinArithmetics, **kwargs: NumericType) -> None:
        """
        Instantiates a _SpinArithmetics instance for the given keyword arguments.
        This class is only defined for qubits, that is all degrees of freedom must 
        have dimension two.

        Arguments:
            `kwargs`: Keyword arguments needed to evaluate, that is access data in,
                the leaf nodes of the operator expression. Leaf nodes are 
                values of type `ElementaryOperator` or `ScalarOperator`.
        """
        self._kwargs = kwargs
