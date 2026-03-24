# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, numpy, warnings
from typing import Sequence

from ..helpers import NumericType
from . import MatrixOperatorTerm, MatrixOperatorElement
from ..scalar import ScalarOperator
import cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime.operators as ops

# Additional operators that are not defined in C++ but have been available
# in cudaq.operators in the past. Hence, we add them here.


def _zero(cls, degree: int):
    return cls("op_zero", [degree])


MatrixOperatorElement.define(
    "op_zero", [0],
    lambda dim: numpy.diag(numpy.zeros(dim, dtype=numpy.complex128)))
MatrixOperatorElement.zero = classmethod(_zero)


def const(constant_value: NumericType) -> ScalarOperator:
    return ScalarOperator.const(constant_value)


def zero(
        degrees: Sequence[int] | int = []
) -> ScalarOperator | MatrixOperatorTerm:
    if hasattr(degrees, "len") and len(degrees) == 0:
        return ScalarOperator.const(0)
    zero_op = MatrixOperatorTerm(0.)
    if isinstance(degrees, int):
        zero_op *= MatrixOperatorTerm(MatrixOperatorElement.zero(degrees))
    else:
        for degree in degrees:
            zero_op *= MatrixOperatorTerm(MatrixOperatorElement.zero(degree))
    return zero_op


def identity(
        degrees: Sequence[int] | int = []
) -> ScalarOperator | MatrixOperatorTerm:
    if hasattr(degrees, "len") and len(degrees) == 0:
        return ScalarOperator.const(1)
    id_op = MatrixOperatorTerm()
    if isinstance(degrees, int):
        id_op *= ops.identity(degrees)
    else:
        for degree in degrees:
            id_op *= ops.identity(degree)
    return id_op


def create(degree: int) -> MatrixOperatorTerm:
    warnings.warn(
        "deprecated - use cudaq.boson.create or cudaq.fermion.create instead, or define your own matrix operator",
        DeprecationWarning)
    return MatrixOperatorTerm(cudaq.boson.create(degree))


def annihilate(degree: int) -> MatrixOperatorTerm:
    warnings.warn(
        "deprecated - use cudaq.boson.annihilate or cudaq.fermion.annihilate instead, or define your own matrix operator",
        DeprecationWarning)
    return MatrixOperatorTerm(cudaq.boson.annihilate(degree))
