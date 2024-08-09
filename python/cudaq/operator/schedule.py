from __future__ import annotations
from collections.abc import Iterator
from typing import Any, Callable, Iterable, Optional

from .helpers import NumericType

class Schedule(Iterator):
    """
    Represents an iterator that produces all values needed for evaluating
    an operator expression at different time steps.
    """

    # The output type of the iterable steps must match the second argument of `get_value`.
    def __init__(self: Schedule, steps: Iterable[Any], parameters: Iterable[str], get_value: Optional[Callable[[str, Any], NumericType]] = None) -> None:
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
        self._iterator = iter(steps)
        self._current_step = None
        self._parameters = parameters
        if get_value is None:
            self._get_value : Callable[[str, Any], NumericType] = self._operator_parameter
        else:
            self._get_value = get_value

    @property
    def _operator_parameter(self: Schedule) -> Callable[[str, NumericType], NumericType]:
        """
        Helper function used in the case when no custom callable to
        retrieve parameter values is defined in the instantiation.
        """
        def resolve_parameter(name: str, value: Any) -> NumericType:
            if name in self._parameters:
                if isinstance(value, (complex, float, int)): return value
                else: raise TypeError("step value is not a numeric type but now function has been defined to compute a numeric type")
            else: raise NotImplementedError(f'unknown parameter {name}')
        return resolve_parameter

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
