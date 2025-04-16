# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


from __future__ import annotations
import inspect, numpy  # type: ignore
from typing import Any, Callable, Mapping, Optional, Tuple
from numpy.typing import NDArray

from .helpers import _OperatorHelpers, NumericType
from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import ScalarOperator as scalar_op_cpp


# FIXME: rather than making this a separate class, extend the Cpp class
class ScalarOperator(scalar_op_cpp):
    """
    Represents a scalar operator defined as a function of zero or more 
    complex-valued parameters.

    Operator expressions cannot be used within quantum kernels, but 
    they provide methods to convert them to data types that can.
    """

    _create_key = object()
    """
    Object used to give an error if a Definition of a scalar operator is
    instantiated outside this class.
    """

    class Definition:
        """
        Represents the definition of a scalar operator.
        """

        __slots__ = ['_generator', '_name']
            
        def __init__(self: ScalarOperator.Definition,
                     generator: Callable[..., NumericType],
                     parameter_info: Optional[Mapping[str, str]],
                     create_key: object) -> None:
            """
            Creates the definition of a scalar operator.
            This constructor should never be called outside the ScalarOperator class.
            
            Arguments:
                generator: See the documentation of the `ScalarOperator.generator` setter.
                parameter_info: A callable that return a mapping of parameter name to 
                    their documentation (if any). If it is set to None, then it will be 
                    created based on the passed generator. 
                
            Note:
            The `parameter_info` passed here can/should be used to accurately propagate
            parameter information for composite scalar operators. It needs to be a callable
            to ensure that the mapping accurately reflects when the generator of a 
            sub-operator is updated.
            """
            assert(create_key == ScalarOperator._create_key), \
                   f"operator definitions must be created using the `{self.__class__.__name__}.generator` setter"
            
            if isinstance(generator, NumericType):
                self._generator = scalar_op_cpp(numpy.complex128(generator))
                self._name = None
                return

            # A variable number of arguments (i.e. `*args`) cannot be supported
            # for generators; it would prevent proper argument handling while
            # supporting additions and multiplication of all kinds of operators.
            arg_spec = inspect.getfullargspec(generator)
            if arg_spec.varargs is not None:
                raise ValueError(
                    f"generator for a '{type(self).__name__}' must not take *args"
                )
            if parameter_info is None:
                parameter_info = {}
                for arg_name in arg_spec.args + arg_spec.kwonlyargs:
                    parameter_info[arg_name] = _OperatorHelpers.parameter_docs(
                        arg_name, generator.__doc__)
            def generator_wrapper(kwargs : dict[str, NumericType]):
                generator_args, remaining_kwargs = _OperatorHelpers.args_from_kwargs(generator, **kwargs)
                return generator(*generator_args, **remaining_kwargs)
            self._generator = scalar_op_cpp(generator_wrapper, **parameter_info)
            self._name = generator.__name__

        @property
        def name(
                self: ScalarOperator.Definition
        ) -> str:
            """
            Contains the name of the generator.
            """
            return self._name

        @property
        def generator(
                self: ScalarOperator.Definition) -> scalar_op_cpp:
            """
            A callable that takes any number of complex-valued keyword arguments and 
            returns the scalar value representing the evaluated operator.
            The parameter names and their documentation (if available) can be accessed
            via the `parameter` property.
            """
            return self._generator

    @classmethod
    def const(cls, constant_value: NumericType) -> ScalarOperator:
        """
        Creates a scalar operator that has a constant value.
        """
        return cls(constant_value)

    __slots__ = ['_definition']

    def __init__(
        self: ScalarOperator,
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
        self._definition = ScalarOperator.Definition(generator, parameter_info,
                                                    self._create_key)
        super(ScalarOperator, self).__init__(self._definition.generator)

    def __eq__(self: ScalarOperator, other: Any) -> bool:
        """
        Returns:
            True, if the other value is a scalar operator with the same generator.
        """
        if type(other) != self.__class__:
            return False
        return self._definition.generator == other._definition.generator

    def _invoke(self: ScalarOperator, **kwargs: NumericType) -> NumericType:
        """
        Helper function that extracts the necessary arguments from the given keyword 
        arguments and invokes the generator. 
        """
        evaluated = self._definition.generator.evaluate(**kwargs)
        if not isinstance(evaluated, NumericType):
            raise TypeError(
                f"generator of {type(self).__name__} must return a number")
        return numpy.complex128(evaluated)

    def evaluate(self: ScalarOperator, **kwargs: NumericType) -> NumericType:
        """
        Invokes the generator with the given keyword arguments.

        Arguments:
            `kwargs`: Keyword arguments needed to evaluate the generator. All
                required parameters and their documentation, if available, can be 
                queried by accessing the `parameter` property.

        Returns:
            The scalar value of the operator for the given keyword arguments.
        """
        return self._invoke(**kwargs)

    # The argument `dimensions` here is only passed for consistency with parent classes.
    def to_matrix(self: ScalarOperator,
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
        return numpy.array([self._invoke(**kwargs)], dtype=numpy.complex128)

    def __str__(self: ScalarOperator) -> str:
        if self._definition.generator.is_constant():
            return str(self._definition.generator.evaluate())
        fct_name = self._definition.name
        if not fct_name or fct_name == "<lambda>": fct_name = "lambda"
        if len(self._definition.generator.parameters) == 0:
            return fct_name
        parameter_names = ", ".join(self.parameters)
        return f"{fct_name}({parameter_names})"

    def _compose_scalar(
        self: ScalarOperator, other: Any,
        fct: Callable[[NumericType, NumericType],
                      NumericType]) -> ScalarOperator:
        """
        Helper function to avoid duplicate code in the various arithmetic 
        operations supported on a ScalarOperator.
        """
        if isinstance(other, NumericType):
            if self._definition.generator.is_constant():
                return ScalarOperator.const(fct(self._definition.generator.evaluate(), other))
            generator = lambda **kwargs: fct(self._invoke(**kwargs), other)
            return ScalarOperator(generator, self._definition.generator.parameters)
        elif type(other) == ScalarOperator:
            if self._definition.generator.is_constant() and other._definition.generator.is_constant():
                return ScalarOperator.const(
                    fct(self._definition.generator.evaluate(), other._definition.generator.evaluate()))
            generator = lambda **kwargs: fct(self._invoke(**kwargs),
                                                other._invoke(**kwargs))
            parameter_info = _OperatorHelpers.aggregate_parameters([
                self._definition.generator.parameters,
                other._definition.generator.parameters
            ])
            return ScalarOperator(generator, parameter_info)
        return NotImplemented

    def _compose(
        self: ScalarOperator, other: Any,
        fct: Callable[[Any, Any], Any]) -> Any:
        """
        Helper function to avoid duplicate code in the various arithmetic 
        operations supported on a ScalarOperator.
        """
        if isinstance(other, NumericType) or type(other) == ScalarOperator:
            return self._compose_scalar(other, fct)
        elif _OperatorHelpers.is_cpp_operator(other):
            return fct(self._definition.generator, other)
        return NotImplemented

    def __pow__(self: ScalarOperator, other: Any) -> Any:
        return self._compose_scalar(other, lambda v1, v2: v1**v2)

    def __mul__(self: ScalarOperator, other: Any) -> Any:
        return self._compose(other, lambda v1, v2: v1 * v2)

    def __truediv__(self: ScalarOperator, other: Any) -> Any:
        return self._compose_scalar(other, lambda v1, v2: v1 / v2)

    def __add__(self: ScalarOperator, other: Any) -> Any:
        return self._compose(other, lambda v1, v2: v1 + v2)

    def __sub__(self: ScalarOperator, other: Any) -> Any:
        return self._compose(other, lambda v1, v2: v1 - v2)

    def __rpow__(self: ScalarOperator, other: Any) -> Any:
        return self._compose(other, lambda v1, v2: v2**v1)

    def __rmul__(self: ScalarOperator, other: Any) -> Any:
        return self._compose(other, lambda v1, v2: v2 * v1)

    def __rtruediv__(self: ScalarOperator, other: Any) -> Any:
        return self._compose(other, lambda v1, v2: v2 / v1)

    def __radd__(self: ScalarOperator, other: Any) -> Any:
        return self._compose(other, lambda v1, v2: v2 + v1)

    def __rsub__(self: ScalarOperator, other: Any) -> Any:
        return self._compose(other, lambda v1, v2: v2 - v1)
