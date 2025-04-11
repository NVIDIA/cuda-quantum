# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
import inspect, numpy  # type: ignore
from typing import Any, Callable, Mapping, Sequence, Tuple
from numpy.typing import NDArray

from .helpers import _OperatorHelpers, NumericType
from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import ElementaryMatrix


class ElementaryOperator:
    """
    Represents an elementary operator that acts on one or more degrees of freedom, 
    and cannot be further simplified to be the product or sum of other operators. 
    An elementary operator is defined as a function of zero or more complex-valued 
    parameters using the :func:`ElementaryOperator.define` class method.

    Operator expressions cannot be used within quantum kernels, but 
    they provide methods to convert them to data types that can.
    """

    _create_key = object()
    """
    Object used to give an error if a Definition of an elementary operator is
    instantiated by other means than the `define` class method.
    """

    class Definition:
        """
        Represents the definition of an elementary operator.
        """

        __slots__ = ['_id', '_expected_dimensions', '_generator']

        def __init__(self: ElementaryOperator.Definition, id: str,
                     expected_dimensions: Sequence[int],
                     create: Callable[..., NDArray[numpy.complexfloating]],
                     create_key: object) -> None:
            """
            Creates the definition of an elementary operator with the given id.
            This constructor should never be called directly, but instead only 
            via the `ElementaryOperator.define` method. See that method for more
            information about the parameters passed here.
            """
            assert(create_key == ElementaryOperator._create_key), \
                   "operator definitions must be created using `ElementaryOperator.define`"
            if len(expected_dimensions) == 0:
                raise ValueError(
                    f"an ElementaryOperator needs to act on at least one degree "
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

            self._id = id
            self._parameters = parameters
            self._expected_dimensions = tuple(expected_dimensions)
            self._generator = lambda dimensions, **kwargs: with_dimension_check(
                create, dimensions, **kwargs)

        @property
        def id(self: ElementaryOperator.Definition) -> str:
            """
            A unique identifier of the operator.
            """
            return self._id

        @property
        def parameters(
                self: ElementaryOperator.Definition) -> Mapping[str, str]:
            """
            Contains information about the parameters required to generate the matrix
            representation of the operator. The keys are the parameter names, 
            and the values are their documentation (if available).
            """
            return self._parameters

        @property
        def expected_dimensions(
                self: ElementaryOperator.Definition) -> Tuple[int]:
            """
            The number of levels, that is the dimension, for each degree of freedom
            in canonical order that the operator acts on. A value of zero or less 
            indicates that the operator is defined for any dimension of that degree.
            """
            return self._expected_dimensions

        @property
        def generator(
            self: ElementaryOperator.Definition
        ) -> Callable[..., NDArray[numpy.complexfloating]]:
            """
            A callable that takes any number of complex-valued keyword arguments and 
            returns the matrix representing the operator in canonical order.
            The parameter names and their documentation (if available) can be accessed
            via the `parameter` property.
            """
            return self._generator

    _ops: dict[str, ElementaryOperator.Definition] = {}
    f"""
    Contains the generator for each defined ElementaryOperator.
    The generator takes a dictionary defining the dimensions for each degree of freedom,
    as well as keyword arguments for complex values.
    """

    @classmethod
    def define(
        cls,
        op_id: str,
        expected_dimensions: Sequence[int],
        create: Callable[..., NDArray[numpy.complexfloating]],
        override: bool = False,
    ) -> None:
        """
        Adds the definition of an elementary operator with the given id to the class.
        After definition, an the defined elementary operator can be instantiated by
        providing the operator id as well as the degree(s) of freedom that it acts on.

        An elementary operator is a parameterized object acting on certain degrees of
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
        if not override and op_id in cls._ops:
            raise ValueError(
                f"an {cls.__name__} with id {op_id} already exists")

        cls._ops[op_id] = cls.Definition(op_id, expected_dimensions, create,
                                         cls._create_key)

    @classmethod
    def zero(cls, degree: int) -> ElementaryOperator:
        op_id = "zero"
        if not op_id in cls._ops:
            cls.define(
                op_id, [0], lambda dim: numpy.zeros(
                    (dim, dim), dtype=numpy.complex128))
        return cls(op_id, (degree,))

    @classmethod
    def identity(cls, degree: int) -> ElementaryOperator:
        op_id = "identity"
        if not op_id in cls._ops:
            cls.define(
                op_id, [0],
                lambda dim: numpy.diag(numpy.ones(dim, dtype=numpy.complex128)))
        return cls(op_id, (degree,))

    __slots__ = ['_id', '_degrees']

    def __init__(self: ElementaryOperator, operator_id: str,
                 degrees: Iterable[int]) -> None:
        """
        Instantiates an elementary operator.

        Arguments:
            operator_id: The id of the operator as specified when it was defined.
            degrees: The degrees of freedom that the operator acts on.
        """
        if not operator_id in ElementaryOperator._ops:
            raise ValueError(
                f"no built-in operator '{operator_id}' has been defined")
        self._id = operator_id
        self._degrees = _OperatorHelpers.canonicalize_degrees(
            degrees
        )  # Sorting so that order matches the elementary operation definition.
        num_degrees = len(self.expected_dimensions)
        if len(degrees) != num_degrees:
            raise ValueError(
                f"definition of {operator_id} acts on {num_degrees} degree(s) of freedom (given: {len(degrees)})"
            )
        super().__init__((self,))

    def __eq__(self: ElementaryOperator, other: Any) -> bool:
        """
        Returns:
            True, if the other value is an elementary operator with the same id
            acting on the same degrees of freedom, and False otherwise.
        """
        if type(other) == self.__class__:
            return self._id == other._id and self._degrees == other._degrees
        # Fallback comparison when the class types are different.
        # This could be the case whereby an elementary operator is the same as a product operator.
        elif self._is_spinop and hasattr(other,
                                         "_is_spinop") and other._is_spinop:
            return self._to_spinop() == other._to_spinop()
        else:
            return False

    @property
    def id(self: ElementaryOperator) -> str:
        """
        A unique identifier of the operator.
        """
        return self._id

    @property
    def parameters(self: ElementaryOperator) -> Mapping[str, str]:
        """
        A mapping that contains the documentation comment for each parameter 
        needed to evaluate the elementary operator.
        """
        return ElementaryOperator._ops[self._id].parameters

    @property
    def expected_dimensions(self: ElementaryOperator) -> Tuple[int]:
        """
        The number of levels, that is the dimension, for each degree of freedom
        in canonical order that the operator acts on. A value of zero or less 
        indicates that the operator is defined for any dimension of that degree.
        """
        return ElementaryOperator._ops[self._id].expected_dimensions

    @property
    def _is_spinop(self: ElementaryOperator) -> bool:
        return self._id in [
            "pauli_x", "pauli_y", "pauli_z", "pauli_i", "identity"
        ]

    def _evaluate(self: ElementaryOperator,
                  arithmetics: OperatorArithmetics[TEval],
                  pad_terms=True) -> TEval:
        """
        Helper function for consistency with other operator expressions.
        Invokes the `evaluate` method of the given OperatorArithmetics.
        """
        return arithmetics.evaluate(self)

    def to_matrix(self: ElementaryOperator, dimensions: Mapping[int, int],
                  **kwargs: NumericType) -> NDArray[numpy.complexfloating]:
        """
        Arguments:
            dimensions: A mapping that specifies the number of levels, that is
                the dimension, of each degree of freedom that the operator acts on.
            `kwargs`: Keyword arguments needed to evaluate the operator. All
                required parameters and their documentation, if available, can be 
                queried by accessing the `parameter` property.

        Returns:
            The matrix representation of the operator expression in canonical order.
        """
        missing_degrees = [degree not in dimensions for degree in self._degrees]
        if any(missing_degrees):
            raise ValueError(
                f'missing dimensions for degree(s) {[self._degrees[i] for i, x in enumerate(missing_degrees) if x]}'
            )
        relevant_dimensions = [dimensions[d] for d in self._degrees]
        return ElementaryOperator._ops[self._id].generator(
            relevant_dimensions, **kwargs)

    def __str__(self: ElementaryOperator) -> str:
        if self._is_spinop:
            # FIXME(OperatorCpp): Currently, the string representation of spin op is used for measurement register name matching, so we need to use the native spin op implementation.
            return str(self._to_spinop())

        parameter_names = ", ".join(self.parameters)
        if parameter_names != "":
            parameter_names = f"({parameter_names})"
        targets = ", ".join((str(degree) for degree in self._degrees))
        return f"{self._id}{parameter_names}[{targets}]"

    def __mul__(self: ElementaryOperator, other: Any) -> ProductOperator:
        if type(other) == ElementaryOperator:
            return ProductOperator((self, other))
        elif isinstance(other, OperatorSum) or isinstance(
                other, (complex, float, int)):
            return ProductOperator((self,)) * other
        return NotImplemented

    def __truediv__(self: ElementaryOperator, other: Any) -> ProductOperator:
        if isinstance(other, (complex, float, int)):
            other_op = ScalarOperator.const(1 / other)
            return ProductOperator((other_op, self))
        if type(other) == ScalarOperator:
            return ProductOperator((1 / other, self))
        return NotImplemented

    def __add__(self: ElementaryOperator, other: Any) -> OperatorSum:
        if type(other) == ElementaryOperator:
            op1 = ProductOperator((self,))
            op2 = ProductOperator((other,))
            return OperatorSum((op1, op2))
        elif isinstance(other, OperatorSum) or isinstance(
                other, (complex, float, int)):
            return OperatorSum((ProductOperator((self,)),)) + other
        return NotImplemented

    def __sub__(self: ElementaryOperator, other: Any) -> OperatorSum:
        return self + (-1. * other)

    def __rmul__(self: ElementaryOperator, other: Any) -> ProductOperator:
        return other * ProductOperator((self,))

    def __radd__(self: ElementaryOperator, other: Any) -> OperatorSum:
        return self + other  # Operator addition is commutative.

    def __rsub__(self: ElementaryOperator, other: Any) -> OperatorSum:
        minus_self = ProductOperator((ScalarOperator.const(-1.), self))
        return minus_self + other  # Operator addition is commutative.
