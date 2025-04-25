# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
import inspect, numpy  # type: ignore
from typing import Callable, Sequence
from numpy.typing import NDArray

from ..helpers import _OperatorHelpers, NumericType
from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import ElementaryOperator, ComplexMatrix


def _defineCustomOperator(cls, id: str,
                expected_dimensions: Sequence[int],
                create: Callable[..., NDArray[numpy.complexfloating]],
                override: bool = False) -> None:
    """
    Creates the definition of an elementary operator with the given id.
    """
    if len(expected_dimensions) == 0:
        raise ValueError(
            f"custom operators needs to act on at least one degree "
            +
            "of freedom - use a ScalarOperator to define operators that "
            + "do not act on any degrees of freedom")

    forwarded_as_kwarg = [["dimensions", "dims"], ["dimension", "dim"]]

    def with_dimension_check(
            creation: Callable, dimensions: Sequence[int],
            **kwargs) -> NDArray[numpy.complexfloating]:
        if any([
                expected > 0 and dimensions[i] != expected
                for i, expected in enumerate(expected_dimensions)
        ]):
            raise ValueError(f'no built-in operator {id} has been defined '\
                            f'for {len(dimensions)} degree(s) of freedom with dimension(s) {dimensions}')
        # If the population of `kwargs` here is changed, adjust the filtering
        # in the `parameters` property below.
        for forwarded in forwarded_as_kwarg[0]:
            kwargs[forwarded] = kwargs.get(
                forwarded, dimensions)  # add if it does not exist
        if len(dimensions) == 1:
            for forwarded in forwarded_as_kwarg[1]:
                kwargs[forwarded] = kwargs.get(
                    forwarded,
                    dimensions[0])  # add if it does not exist
        creation_args, remaining_kwargs = _OperatorHelpers.args_from_kwargs(
            creation, **kwargs)
        evaluated = creation(*creation_args, **remaining_kwargs)
        if not isinstance(evaluated, numpy.ndarray):
            raise TypeError(
                "operator evaluation must return a NDArray[complex]")
        return evaluated

    parameters, forwarded = {}, [
        keyword for group in forwarded_as_kwarg for keyword in group
    ]
    arg_spec = inspect.getfullargspec(create)
    for pname in arg_spec.args + arg_spec.kwonlyargs:
        if not pname in forwarded:
            parameters[pname] = _OperatorHelpers.parameter_docs(
                pname, create.__doc__)

    def generator_wrapper(dimensions: Sequence[int], kwargs: dict[str, NumericType]):
        np_matrix = with_dimension_check(create, dimensions, **kwargs)
        return ComplexMatrix(np_matrix)
    cls._define(id, expected_dimensions, generator_wrapper, override, **parameters)

ElementaryOperator.define = classmethod(_defineCustomOperator)


