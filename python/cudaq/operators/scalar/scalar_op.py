# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
import numpy  # type: ignore
from typing import Any, Callable, Mapping
from numpy.typing import NDArray

from ..helpers import NumericType, _aggregate_parameters
from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import ScalarOperator


def _const_init(cls, constant_value: NumericType) -> ScalarOperator:
    """
    Creates a scalar operator that has a constant value.
    """
    if not isinstance(constant_value, NumericType):
        raise ValueError("argument must be a numeric constant")
    return cls(constant_value)


ScalarOperator.const = classmethod(_const_init)


# The argument `dimensions` here is only passed for consistency with parent classes.
def _to_matrix(self: ScalarOperator,
               dimensions: Mapping[int, int] = {},
               **kwargs: NumericType) -> NDArray[numpy.complexfloating]:
    """
    Class method for consistency with other operator classes.
    Invokes the generator with the given keyword arguments.

    Arguments:
        dimensions: (unused, passed for consistency) 
            A mapping that specifies the number of levels, that is
            the dimension, of each degree of freedom that the operator acts on.
        `kwargs`: Keyword arguments needed to evaluate the generator. All
            required parameters and their documentation, if available, can be 
            queried by accessing the `parameter` property.

    Returns:
        An array with a single element corresponding to the value of the operator 
        for the given keyword arguments.
    """
    return numpy.array([self.evaluate(**kwargs)], dtype=numpy.complex128)


ScalarOperator.to_matrix = _to_matrix


def _compose(
        self: ScalarOperator, other: Any,
        fct: Callable[[NumericType, NumericType],
                      NumericType]) -> ScalarOperator:
    """
    Helper function to avoid duplicate code in the various arithmetic 
    operations supported on a ScalarOperator.
    """
    if isinstance(other, NumericType):
        if self.is_constant():
            return ScalarOperator.const(fct(self.evaluate(), other))
        generator = lambda **kwargs: fct(self.evaluate(**kwargs), other)
        return ScalarOperator(generator, self.parameters)
    elif type(other) == ScalarOperator:
        if self.is_constant() and other.is_constant():
            return ScalarOperator.const(fct(self.evaluate(), other.evaluate()))
        generator = lambda **kwargs: fct(self.evaluate(**kwargs),
                                         other.evaluate(**kwargs))
        parameter_info = _aggregate_parameters(
            [self.parameters, other.parameters])
        return ScalarOperator(generator, parameter_info)
    return NotImplemented


ScalarOperator.__pow__ = lambda self, other: _compose(self, other, lambda v1,
                                                      v2: v1**v2)
ScalarOperator.__mul__ = lambda self, other: _compose(self, other, lambda v1,
                                                      v2: v1 * v2)
ScalarOperator.__truediv__ = lambda self, other: _compose(
    self, other, lambda v1, v2: v1 / v2)
ScalarOperator.__add__ = lambda self, other: _compose(self, other, lambda v1,
                                                      v2: v1 + v2)
ScalarOperator.__sub__ = lambda self, other: _compose(self, other, lambda v1,
                                                      v2: v1 - v2)
ScalarOperator.__rpow__ = lambda self, other: _compose(self, other, lambda v1,
                                                       v2: v2**v1)
ScalarOperator.__rmul__ = lambda self, other: _compose(self, other, lambda v1,
                                                       v2: v2 * v1)
ScalarOperator.__rtruediv__ = lambda self, other: _compose(
    self, other, lambda v1, v2: v2 / v1)
ScalarOperator.__radd__ = lambda self, other: _compose(self, other, lambda v1,
                                                       v2: v2 + v1)
ScalarOperator.__rsub__ = lambda self, other: _compose(self, other, lambda v1,
                                                       v2: v2 - v1)
