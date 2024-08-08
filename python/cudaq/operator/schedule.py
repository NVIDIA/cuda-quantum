from __future__ import annotations
from collections import Iterator
from typing import Any, Callable, Iterable, Optional

from .helpers import NumericType
from .mlir._mlir_libs._quakeDialects import cudaq_runtime

class Schedule(Iterator):
    """
    Represents an iterator that produces all values needed for evaluating
    an operator expression at different time steps.
    """

    # The output type of the iterable steps must match the second argument of `get_value`.
    def __init__(self: Schedule, steps: Iterable[Any], parameters: Iterable[str], get_value: Callable[[str, Any], NumericType]) -> None:
        """
        Creates a schedule for evaluating an operator expression at different steps.

        Arguments:
            steps: The sequence of steps in the schedule. A step is defined as a value 
                of arbitrary type.
            parameters: A sequence of strings representing the parameter names of an 
                operator expression.
            get_value: A function that takes the name of a parameter as well as an 
                additional value ("step") of arbitrary type as argument and returns the 
                complex value for that parameter at the given step.
        """
        self._iterator = iter(steps)
        self._parameters = parameters
        self._get_value = get_value
        self._current_step = None

    @property
    def current_step(self: Schedule) -> Optional[Any]:
        """
        The value of the step the Schedule (iterator) is currently at.
        """
        return self._current_step
    
    def __iter__(self: Schedule) -> Schedule:
        return self
        
    def __next__(self: Schedule) -> Mapping[str, NumericType]:
        self._current_step = next(self._iterator)
        kwargs : dict[str, NumericType] = {}
        for parameter in self._parameters:
            kwargs[parameter] = self._get_value(parameter, self._current_step)
        return kwargs
