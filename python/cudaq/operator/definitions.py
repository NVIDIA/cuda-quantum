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

from .helpers import NumericType, _OperatorHelpers
from .expressions import OperatorSum, ProductOperator, ElementaryOperator, ScalarOperator
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
            """Connects to the next available port.
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
            """Connects to the next available port.
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


class pauli:
    ElementaryOperator.define(
        "pauli_x", [2], lambda: _OperatorHelpers.cmatrix_to_nparray(
            cudaq_runtime.spin.x(0).to_matrix()))
    ElementaryOperator.define(
        "pauli_y", [2], lambda: _OperatorHelpers.cmatrix_to_nparray(
            cudaq_runtime.spin.y(0).to_matrix()))
    ElementaryOperator.define(
        "pauli_z", [2], lambda: _OperatorHelpers.cmatrix_to_nparray(
            cudaq_runtime.spin.z(0).to_matrix()))

    @classmethod
    def x(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_x", [degree])

    @classmethod
    def y(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_y", [degree])

    @classmethod
    def z(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_z", [degree])

    @classmethod
    def i(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator.identity(degree)

    @classmethod
    def plus(cls, degree: int) -> OperatorSum:
        return (cls.x(degree) + ScalarOperator.const(1j) * cls.y(degree)) / 2

    @classmethod
    def minus(cls, degree: int) -> OperatorSum:
        return (cls.x(degree) - ScalarOperator.const(1j) * cls.y(degree)) / 2
