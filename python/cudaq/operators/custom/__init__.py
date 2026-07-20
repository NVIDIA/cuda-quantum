# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import typing
import numpy.typing
from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime.operators import *
from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import MatrixOperator, MatrixOperatorTerm
from .custom_op import MatrixOperatorElement
from .backwards_compatibility import *


def define(id: str,
           expected_dimensions: Sequence[int],
           create: typing.Callable[...,
                                   numpy.typing.NDArray[numpy.complexfloating]],
           override: bool = False) -> None:
    """
    Defines a matrix operator element with the given id.
    After definition, an the defined elementary operator can be instantiated by
    providing the operator id as well as the degree(s) of freedom that it acts on.
    A matrix operator element is a parameterized object acting on certain degrees of
    freedom. To evaluate an operator, for example to compute its matrix, the level,
    that is the dimension, for each degree of freedom it acts on must be provided,
    as well as all additional parameters. Additional parameters must be provided in
    the form of keyword arguments.

    Note:
    The dimensions passed during operator evaluation are automatically validated
    against the expected dimensions specified during definition - the `create`
    function does not need to do this.

    Arguments:
        op_id: A string that uniquely identifies the defined operator.
        expected_dimensions: defines the number of levels, that is the dimension,
            for each degree of freedom in canonical (that is sorted) order. A
            negative or zero value for one (or more) of the expected dimensions
            indicates that the operator is defined for any dimension of the
            corresponding degree of freedom.
        create: Takes any number of complex-valued arguments and returns the
            matrix representing the operator in canonical order. If the matrix can
            be defined for any number of levels for one or more degree of freedom,
            the `create` function must take an argument called `dimension` (or `dim`
            for short), if the operator acts on a single degree of freedom, and an
            argument called `dimensions` (or `dims` for short), if the operator acts
            on multiple degrees of freedom.
        override: if True it allows override the definition. (default: False)
    """
    MatrixOperatorElement.define(id, expected_dimensions, create, override)


def instantiate(op_id: str,
                degrees: int | typing.Iterable[int]) -> MatrixOperatorTerm:
    """
    Instantiates a product operator containing a previously defined operator element.

    Arguments:
        operator_id: The id of the operator element as specified when it was defined.
        degrees: The degree(s) of freedom that the operator acts on.
    """
    if isinstance(degrees, int):
        degrees = [degrees]
    element = MatrixOperatorElement(op_id, degrees)
    return MatrixOperatorTerm(element)
