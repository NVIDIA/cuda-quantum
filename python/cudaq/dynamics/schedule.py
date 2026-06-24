# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from typing import Any, Callable, Iterable, Iterator, Optional

from ..operators import NumericType


class Schedule(Iterator):
    """
    Represents an iterator that produces all values needed for evaluating
    an operator expression at different time steps.
    """

    # The type of the steps sequence must match the second argument of `get_value`.
    __slots__ = ['_steps', '_current_idx', '_parameters', '_get_value']

    def __init__(
            self: Schedule,
            steps: Iterable[Any],
            parameters: Iterable[str],
            get_value: Optional[Callable[[str, Any],
                                         NumericType]] = None) -> None:
        """
        Creates a schedule for evaluating an operator expression at different steps.

        Arguments:
            steps: The sequence of steps in the schedule. A step is defined as a value 
                of arbitrary type.
            parameters: A sequence of strings representing the parameter names of an 
                operator expression.
            get_value: A function that takes the name of a parameter as well as an 
                additional value ("step") of arbitrary type as argument and returns the 
                complex value for that parameter at the given step. If this function is 
                not provided, then the steps must be of a numeric type, and the value
                of each parameter will be set to the step value. 
        """
        self._steps = tuple(steps)
        self._current_idx = -1
        self._parameters = tuple(parameters)
        if get_value is None:
            self._get_value: Callable[[str, Any],
                                      NumericType] = self._operator_parameter
        else:
            self._get_value = get_value

    @property
    def _operator_parameter(
            self: Schedule) -> Callable[[str, NumericType], NumericType]:
        """
        Helper function used in the case when no custom callable to
        retrieve parameter values is defined in the instantiation.
        """

        def resolve_parameter(name: str, value: Any) -> NumericType:
            if name in self._parameters:
                if isinstance(value, (complex, float, int)):
                    return value
                else:
                    raise TypeError(
                        "step value is not a numeric type but now function has been defined to compute a numeric type"
                    )
            else:
                raise NotImplementedError(f'unknown parameter {name}')

        return resolve_parameter

    @property
    def current_step(self: Schedule) -> Optional[Any]:
        """
        The value of the step the Schedule (iterator) is currently at.
        Returns None if the iteration has not yet started or has finished.
        """
        if 0 <= self._current_idx < len(self._steps):
            return self._steps[self._current_idx]
        else:
            return None

    @property
    def next_step(self: Schedule) -> Optional[Any]:
        """
        The value of the next step of the current Schedule.
        Returns None if the iteration has finished.
        """
        if 0 <= self._current_idx < len(self._steps) - 1:
            return self._steps[self._current_idx + 1]
        else:
            return None

    def __len__(self):
        return len(self._steps)

    def reset(self: Schedule) -> None:
        """
        Resets the schedule (iterator) to its starting point.
        """
        self._current_idx = -1

    def __iter__(self: Schedule) -> Schedule:
        return self

    def __next__(self: Schedule) -> Mapping[str, NumericType]:
        self._current_idx += 1
        current_step = self.current_step
        if current_step is None:
            raise StopIteration
        return dict(((parameter, self._get_value(parameter, current_step))
                     for parameter in self._parameters))
