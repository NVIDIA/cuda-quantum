# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from typing import Sequence
from cudaq import boson, fermion, spin_op as spin, ops

from .helpers import NumericType
from .expressions import RydbergHamiltonian

from ..spin_op import SpinOperator, SpinOperatorTerm
from ..fermion import FermionOperator, FermionOperatorTerm
from ..boson import BosonOperator, BosonOperatorTerm
from ..ops import MatrixOperator, MatrixOperatorTerm
from .scalar_op import ScalarOperator
from .custom_op import ElementaryOperator

class operators:

    @classmethod
    def const(cls, constant_value: NumericType) -> ScalarOperator:
        return ScalarOperator.const(constant_value)

    @classmethod
    def zero(
        cls,
        degrees: Sequence[int] | int = []
    ) -> ScalarOperator | MatrixOperatorTerm:
        if len(degrees) == 0:
            return ScalarOperator.const(0)
        zero_op = MatrixOperatorTerm(0.)
        if isinstance(degrees, int):
            zero_op *= ops.identity(degrees)
        else:
            for degree in degrees:
                zero_op *= ops.identity(degree)
        return zero_op

    @classmethod
    def identity(
        cls,
        degrees: Sequence[int] | int = []
    ) -> ScalarOperator | MatrixOperatorTerm:
        if len(degrees) == 0:
            return ScalarOperator.const(1)
        id_op = MatrixOperatorTerm()
        if isinstance(degrees, int):
            id_op *= ops.identity(degrees)
        else:
            for degree in degrees:
                id_op *= ops.identity(degree)
        return id_op

    @classmethod
    def create(cls, degree: int) -> BosonOperatorTerm:
        return boson.create(degree)

    @classmethod
    def annihilate(cls, degree: int) -> BosonOperatorTerm:
        return boson.annihilate(degree)

    @classmethod
    def number(cls, degree: int) -> BosonOperatorTerm:
        return boson.number(degree)

    @classmethod
    def parity(cls, degree: int) -> MatrixOperatorTerm:
        return ops.parity(degree)

    @classmethod
    def displace(cls, degree: int) -> MatrixOperatorTerm:
        return ops.displace(degree)

    @classmethod
    def squeeze(cls, degree: int) -> MatrixOperatorTerm:
        return ops.squeeze(degree)

    @classmethod
    def position(cls, degree: int) -> BosonOperatorTerm:
        return boson.position(degree)

    @classmethod
    def momentum(cls, degree: int) -> BosonOperatorTerm:
        return boson.momentum(degree)

