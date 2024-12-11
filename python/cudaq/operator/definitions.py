# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy, scipy  # type: ignore
from numpy.typing import NDArray
from typing import Sequence

from .helpers import NumericType
from .expressions import OperatorSum, ProductOperator, ElementaryOperator, ScalarOperator, RydbergHamiltonian
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime


# Operators as defined here (watch out of differences in convention):
# https://www.dynamiqs.org/python_api/utils/operators/sigmay.html
class operators:

    class matrices:

        @staticmethod
        def _create(dimension: int) -> NDArray[numpy.complexfloating]:
            return numpy.diag(
                numpy.sqrt(numpy.arange(1, dimension, dtype=numpy.complex128)),
                -1)

        @staticmethod
        def _annihilate(dimension: int) -> NDArray[numpy.complexfloating]:
            return numpy.diag(
                numpy.sqrt(numpy.arange(1, dimension, dtype=numpy.complex128)),
                1)

        @staticmethod
        def _position(dimension: int) -> NDArray[numpy.complexfloating]:
            return complex(0.5) * (operators.matrices._create(dimension) +
                                   operators.matrices._annihilate(dimension))

        @staticmethod
        def _momentum(dimension: int) -> NDArray[numpy.complexfloating]:
            return 0.5j * (operators.matrices._create(dimension) -
                           operators.matrices._annihilate(dimension))

        @staticmethod
        def _displace(
                dimension: int,
                displacement: NumericType) -> NDArray[numpy.complexfloating]:
            """Returns the displacement operator matrix.
            Args:
                displacement: Amplitude of the displacement operator.
                    See also https://en.wikipedia.org/wiki/Displacement_operator.
            """
            displacement = complex(displacement)
            term1 = displacement * operators.matrices._create(dimension)
            term2 = numpy.conjugate(
                displacement) * operators.matrices._annihilate(dimension)
            return scipy.linalg.expm(term1 - term2)

        @staticmethod
        def _squeeze(dimension: int,
                     squeezing: NumericType) -> NDArray[numpy.complexfloating]:
            """Returns the squeezing operator matrix.
            Args:
                squeezing: Amplitude of the squeezing operator.
                    See also https://en.wikipedia.org/wiki/Squeeze_operator.
            """
            squeezing = complex(squeezing)
            term1 = numpy.conjugate(squeezing) * numpy.linalg.matrix_power(
                operators.matrices._annihilate(dimension), 2)
            term2 = squeezing * numpy.linalg.matrix_power(
                operators.matrices._create(dimension), 2)
            return scipy.linalg.expm(0.5 * (term1 - term2))

    ElementaryOperator.define("op_create", [0], matrices._create)
    ElementaryOperator.define("op_annihilate", [0], matrices._annihilate)
    ElementaryOperator.define(
        "op_number", [0],
        lambda dim: numpy.diag(numpy.arange(dim, dtype=numpy.complex128)))
    ElementaryOperator.define(
        "op_parity", [0],
        lambda dim: numpy.diag([(-1. + 0j)**i for i in range(dim)]))
    ElementaryOperator.define("op_displace", [0], matrices._displace)
    ElementaryOperator.define("op_squeeze", [0], matrices._squeeze)
    ElementaryOperator.define("op_position", [0], matrices._position)
    ElementaryOperator.define("op_momentum", [0], matrices._momentum)

    @classmethod
    def const(cls, constant_value: NumericType) -> ScalarOperator:
        return ScalarOperator.const(constant_value)

    @classmethod
    def zero(
        cls,
        degrees: Sequence[int] | int = []
    ) -> ScalarOperator | ElementaryOperator | ProductOperator:
        if isinstance(degrees, int):
            return ElementaryOperator.zero(degrees)
        elif len(degrees) == 0:
            return ScalarOperator.const(0)
        elif len(degrees) == 1:
            return ElementaryOperator.zero(degrees[0])
        else:
            return ProductOperator(
                [ElementaryOperator.zero(degree) for degree in degrees])

    @classmethod
    def identity(
        cls,
        degrees: Sequence[int] | int = []
    ) -> ScalarOperator | ElementaryOperator | ProductOperator:
        if isinstance(degrees, int):
            return ElementaryOperator.identity(degrees)
        elif len(degrees) == 0:
            return ScalarOperator.const(1)
        elif len(degrees) == 1:
            return ElementaryOperator.identity(degrees[0])
        else:
            return ProductOperator(
                [ElementaryOperator.identity(degree) for degree in degrees])

    @classmethod
    def create(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_create", [degree])

    @classmethod
    def annihilate(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_annihilate", [degree])

    @classmethod
    def number(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_number", [degree])

    @classmethod
    def parity(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_parity", [degree])

    @classmethod
    def displace(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_displace", [degree])

    @classmethod
    def squeeze(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_squeeze", [degree])

    @classmethod
    def position(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_position", [degree])

    @classmethod
    def momentum(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_momentum", [degree])


class spin:
    ElementaryOperator.define(
        "pauli_x", [2], lambda: cudaq_runtime.spin.x(0).to_matrix().to_numpy())
    ElementaryOperator.define(
        "pauli_y", [2], lambda: cudaq_runtime.spin.y(0).to_matrix().to_numpy())
    ElementaryOperator.define(
        "pauli_z", [2], lambda: cudaq_runtime.spin.z(0).to_matrix().to_numpy())

    @classmethod
    def x(cls, target: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_x", [target])

    @classmethod
    def y(cls, target: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_y", [target])

    @classmethod
    def z(cls, target: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_z", [target])

    @classmethod
    def i(cls, target: int) -> ElementaryOperator:
        return ElementaryOperator.identity(target)

    @classmethod
    def plus(cls, degree: int) -> OperatorSum:
        return (cls.x(degree) + ScalarOperator.const(1j) * cls.y(degree)) / 2

    @classmethod
    def minus(cls, degree: int) -> OperatorSum:
        return (cls.x(degree) - ScalarOperator.const(1j) * cls.y(degree)) / 2


# Trampoline class to maintain backward compatibility with native `SpinOperator` class.
# In particular, it dispatches static methods, e.g., `random()` as well as various factory methods, e.g., create from file, serialized data, etc.
class SpinOperator(OperatorSum):

    def __init__(self):
        # This should never be called. We have `__new__` method instead.
        raise ValueError("Not supported")

    # Convert from a Pauli word to an Operator
    @staticmethod
    def from_word(word: str) -> ProductOperator:
        """
        Return a :class:`SpinOperator` corresponding to the provided Pauli `word`.

        ```
            # Example:
            # The first and third qubits will receive a Pauli X,
            # while the second qubit will receive a Pauli Y.
            word = "XYX"
            # Convert word to spin operator.
            spin_operator = cudaq.SpinOperator.from_word(word)
            print(spin_operator) # prints: `[1+0j] XYX`)#")
        ```
        """

        return ProductOperator._from_word(word)

    def __new__(cls, *args, **kwargs):
        if len(kwargs) == 0 and len(args) == 0:
            # This is a legacy behavior: `SpinOperator()` returns an identity term.
            return ElementaryOperator.identity(0)
        # Handle copy constructor
        if len(args) == 1 and hasattr(args[0], "_to_spinop"):
            return OperatorSum._from_spin_op(
                cudaq_runtime.SpinOperator(spin_operator=args[0]._to_spinop()))
        if "spin_operator" in kwargs and hasattr(kwargs["spin_operator"],
                                                 "_to_spinop"):
            return OperatorSum._from_spin_op(
                cudaq_runtime.SpinOperator(
                    spin_operator=kwargs["spin_operator"]._to_spinop()))
        # For all other constructors: e.g., from serialized data, file, `OpenFermion` object,
        # forward it to the runtime implementation and convert back to the new operator class.
        # FIXME(OperatorCpp): Remove this when the operator class is implemented in C++
        return OperatorSum._from_spin_op(
            cudaq_runtime.SpinOperator(*args, **kwargs))

    @staticmethod
    def random(qubit_count: int, term_count: int, seed: int | None = None):
        """
        Return a random `SpinOperator` on the given number of qubits (`qubit_count`) and composed of the given number of terms (`term_count`). An optional seed value may also be provided. 
        """

        # FIXME(OperatorCpp): The logic of this `SpinOperator` random is specific and unit-tested.
        # So, we do a conversion to guarantee compatibility.
        if seed is None:
            return OperatorSum._from_spin_op(
                cudaq_runtime.SpinOperator.random(qubit_count, term_count))
        else:
            return OperatorSum._from_spin_op(
                cudaq_runtime.SpinOperator.random(qubit_count, term_count,
                                                  seed))
