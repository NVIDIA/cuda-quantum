# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
import inspect, numpy  # type: ignore
from typing import Any, Callable, Mapping, Optional
from numpy.typing import NDArray

from ..helpers import NumericType, _aggregate_parameters, _args_from_kwargs, _parameter_docs
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


def _instantiate(cls,
                 generator: NumericType | Callable[..., NumericType],
                 parameter_info: Optional[Mapping[str, str]] = None) -> None:
    """
    Instantiates a scalar operator.

    Arguments:
        generator: The value of the scalar operator as a function of its
            parameters. The generator may take any number of complex-valued
            arguments and must return a number. Each parameter must be passed
            as a keyword argument when evaluating the operator. 
    """
    instance = super(ScalarOperator, cls).__new__(cls)
    if isinstance(generator, NumericType):
        instance.__init__(numpy.complex128(generator))
    else:
        # A variable number of arguments (i.e. `*args`) cannot be supported
        # for generators; it would prevent proper argument handling while
        # supporting additions and multiplication of all kinds of operators.
        arg_spec = inspect.getfullargspec(generator)
        if arg_spec.varargs is not None:
            raise ValueError(
                f"the function defining a scalar operator must not take *args")
        if parameter_info is None:
            parameter_info = {}
            for arg_name in arg_spec.args + arg_spec.kwonlyargs:
                parameter_info[arg_name] = _parameter_docs(
                    arg_name, generator.__doc__)

        def generator_wrapper(kwargs: dict[str, NumericType]):
            generator_args, remaining_kwargs = _args_from_kwargs(
                generator, **kwargs)
            return generator(*generator_args, **remaining_kwargs)

        instance.__init__(generator_wrapper, **parameter_info)
    return instance


ScalarOperator.__new__ = staticmethod(_instantiate)
