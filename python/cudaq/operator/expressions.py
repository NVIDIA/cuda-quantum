# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
import inspect, math, numpy  # type: ignore
from typing import Any, Callable, Generator, Iterable, Mapping, Optional, Sequence
from numpy.typing import NDArray

from .helpers import _OperatorHelpers, NumericType
from .manipulation import MatrixArithmetics, OperatorArithmetics, PrettyPrint, _SpinArithmetics
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime


class OperatorSum:
    """
    Represents an operator expression consisting of a sum of terms, 
    where each term is a product of elementary and scalar operators. 

    Operator expressions cannot be used within quantum kernels, but 
    they provide methods to convert them to data types that can.
    """

    # FIXME: implement a caching mechanism for to_matrix

    __slots__ = ['_terms']

    def __init__(self: OperatorSum,
                 terms: Iterable[ProductOperator] = []) -> None:
        """
        Instantiates an operator expression that represents a sum of terms,
        where each term is a product of elementary and scalar operators. 

        Arguments:
            terms: The `ProductOperators` that should be summed up when 
                evaluating the operator expression.
        """
        self._terms = tuple(terms)
        if len(self._terms) == 0:
            self._terms = (ProductOperator((ScalarOperator.const(0),)),)

    def _canonical_terms(
        self: OperatorSum
    ) -> Sequence[tuple[ScalarOperator | ElementaryOperator]]:
        """
        Helper function used to compute the operator hash and equality.
        """

        def canonicalize_product(
            prod: ProductOperator
        ) -> tuple[ScalarOperator | ElementaryOperator]:
            all_degrees = [
                degree for op in prod._operators for degree in op._degrees
            ]
            scalars = [
                op for op in prod._operators if isinstance(op, ScalarOperator)
            ]
            non_scalars = [
                op for op in prod._operators
                if not isinstance(op, ScalarOperator)
            ]
            if len(all_degrees) == len(frozenset(all_degrees)):
                # Each operator acts on different degrees of freedom; they
                # hence commute and can be reordered arbitrarily.
                non_scalars.sort(key=lambda op: op._degrees)
            else:
                # Some degrees exist multiple times; order the scalars, identities and zeros,
                # but do not otherwise try to reorder terms.
                zero_ops = (op for op in non_scalars if op._id == "zero")
                identity_ops = (
                    op for op in non_scalars if op._id == "identity")
                non_commuting = (op for op in non_scalars
                                 if op._id != "zero" and op._id != "identity")
                non_scalars = [
                    *sorted(zero_ops, key=lambda op: op._degrees),
                    *sorted(identity_ops, key=lambda op: op._degrees),
                    *non_commuting
                ]
            if len(scalars) > 1:
                return (math.prod(scalars), *non_scalars)
            else:
                return (*scalars, *non_scalars)

        # Operator addition is commutative and terms can hence be arbitrarily reordered.
        return sorted((canonicalize_product(term) for term in self._terms),
                      key=lambda ops: str(ProductOperator(ops)))

    def canonicalize(self: OperatorSum) -> OperatorSum:
        f"""
        Creates a new {self.__class__.__name__} instance where all sub-terms are 
        sorted in canonical order. The new instance is equivalent to the original one, 
        meaning it has the same effect on any quantum system for any set of parameters. 
        """
        canonical_terms = (
            ProductOperator(operators) for operators in self._canonical_terms())
        return OperatorSum(canonical_terms)

    def __eq__(self: OperatorSum, other: Any) -> bool:
        f"""
        Returns:
            True, if the other value is an {self.__class__.__name__} with equivalent 
            terms, and False otherwise. The equality takes into account that operator
            addition is commutative, as is the product of two operators if they act 
            on different degrees of freedom.
            The equality comparison does *not* take commutation relations into 
            account, and does not try to reorder terms block-wise; it may hence 
            evaluate to False, even if two operators in reality are the same.
            If the equality evaluates to True, on the other hand, the operators 
            are guaranteed to represent the same transformation for all arguments.
        """
        return type(other) == self.__class__ and self._canonical_terms(
        ) == other._canonical_terms()

    @property
    def degrees(self: OperatorSum) -> tuple[int]:
        """
        The degrees of freedom that the operator acts on in canonical order.
        """
        unique_degrees = frozenset((degree for term in self._terms
                                    for op in term._operators
                                    for degree in op._degrees))
        # Sorted in canonical order to match the to_matrix method.
        return _OperatorHelpers.canonicalize_degrees(unique_degrees)

    @property
    def parameters(self: OperatorSum) -> Mapping[str, str]:
        """
        A mapping that contains the documentation comment for each parameter 
        needed to evaluate the operator expression.
        """
        return _OperatorHelpers.aggregate_parameters(
            (op.parameters for term in self._terms for op in term._operators))

    @property
    def _is_spinop(self: OperatorSum) -> bool:
        # FIXME: Make spin operators inherit from operators
        return all(
            (op._is_spinop for term in self._terms for op in term._operators))

    def _evaluate(self: OperatorSum,
                  arithmetics: OperatorArithmetics[TEval]) -> TEval:
        """
        Helper function used for evaluating operator expressions and computing arbitrary values
        during evaluation. The value to be computed is defined by the OperatorArithmetics.
        The evaluation guarantees that addition and multiplication of two operators will only
        be called when both operators act on the same degrees of freedom, and the tensor product
        will only be computed if they act on different degrees of freedom. 
        """
        degrees = frozenset((degree for term in self._terms
                             for op in term._operators
                             for degree in op._degrees))

        # We need to make sure all matrices are of the same size to sum them up.
        def padded_term(term: ProductOperator) -> ProductOperator:
            op_degrees = [
                op_degree for op in term._operators for op_degree in op._degrees
            ]
            for degree in degrees:
                if not degree in op_degrees:
                    term *= ElementaryOperator.identity(degree)
            return term

        sum = padded_term(self._terms[0])._evaluate(arithmetics)
        for term in self._terms[1:]:
            sum = arithmetics.add(sum, padded_term(term)._evaluate(arithmetics))
        return sum

    def to_matrix(self: OperatorSum, dimensions: Mapping[int, int],
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
        # FIXME: make sure we have enough tests that are not consisting of spin operators,
        # then enable the code below and check tests with and without this.
        # FIXME: while the convention (`endianness`) in operators should match the one in
        # the rest of CUDA-Q, there are some bugs in `spinop` arithmetic that need to be fixed.
        # For example, while `pauliX(0) * (pauliZ(0) + pauliZ(1))` evaluates correctly,
        # this does not: `(pauliZ(0) + pauliZ(1)) * pauliX(0)`
        return self._evaluate(MatrixArithmetics(dimensions, **kwargs)).matrix

    # To be removed/replaced. We need to be able to pass general operators to cudaq.observe.
    def _to_spinop(
            self: OperatorSum, dimensions: Mapping[int, int],
            **kwargs: NumericType) -> cudaq_runtime.SpinOperator | NumericType:
        if any((dim != 2 for dim in dimensions.values())):
            raise ValueError(
                "incorrect dimensions - conversion to spin operator can only be done for qubits"
            )
        return self._evaluate(_SpinArithmetics(**kwargs))

    def to_pauli_word(self: OperatorSum) -> cudaq_runtime.pauli_word:
        """
        Creates a representation of the operator as `pauli_word` that can be passed
        as an argument to quantum kernels.
        Raises a ValueError if the operator contains non-Pauli sub-operators.
        """
        return self._evaluate(PauliWordConversion()).pauli_word

    def __str__(self: OperatorSum) -> str:
        return self._evaluate(PrettyPrint())

    def __mul__(self: OperatorSum, other: Any) -> OperatorSum:
        if type(other) == OperatorSum:
            return OperatorSum((self_term * other_term
                                for self_term in self._terms
                                for other_term in other._terms))
        elif isinstance(other, OperatorSum) or isinstance(
                other, (complex, float, int)):
            return OperatorSum((self_term * other for self_term in self._terms))
        return NotImplemented

    def __truediv__(self: OperatorSum, other: Any) -> OperatorSum:
        if type(other) == ScalarOperator or isinstance(other,
                                                       (complex, float, int)):
            return OperatorSum((term / other for term in self._terms))
        return NotImplemented

    def __add__(self: OperatorSum, other: Any) -> OperatorSum:
        if type(other) == OperatorSum:
            return OperatorSum((*self._terms, *other._terms))
        elif isinstance(other, (complex, float, int)):
            other_term = ProductOperator((ScalarOperator.const(other),))
            return OperatorSum((*self._terms, other_term))
        elif type(
                other
        ) == ScalarOperator:  # Elementary and product operators are handled by their classes.
            return OperatorSum((*self._terms, ProductOperator((other,))))
        return NotImplemented

    def __sub__(self: OperatorSum, other: Any) -> OperatorSum:
        return self + (-1 * other)

    def __rmul__(self: OperatorSum, other: Any) -> OperatorSum:
        if isinstance(other, (complex, float, int)):
            return OperatorSum((self_term * other for self_term in self._terms))
        elif type(other) == ProductOperator:
            return OperatorSum((other,)) * self
        elif type(other) == ScalarOperator or type(other) == ElementaryOperator:
            return OperatorSum((ProductOperator((other,)),)) * self
        return NotImplemented

    def __radd__(self: OperatorSum, other: Any) -> OperatorSum:
        if isinstance(other, (complex, float, int)):
            other_term = ProductOperator((ScalarOperator.const(other),))
            return OperatorSum((other_term, *self._terms))
        elif type(
                other
        ) == ScalarOperator:  # Elementary and product operators are handled by their classes.
            return OperatorSum((ProductOperator((other,)), *self._terms))
        return NotImplemented

    def __rsub__(self: OperatorSum, other: Any) -> OperatorSum:
        return (-1 * self) + other  # Operator addition is commutative.


class ProductOperator(OperatorSum):
    """
    Represents an operator expression consisting of a product of elementary
    and scalar operators. 

    Operator expressions cannot be used within quantum kernels, but 
    they provide methods to convert them to data types that can.
    """

    __slots__ = ['_operators']

    def __init__(
        self: ProductOperator,
        atomic_operators: Iterable[ElementaryOperator | ScalarOperator] = []
    ) -> None:
        """
        Instantiates an operator expression that represents a product of elementary
        or scalar operators.

        Arguments:
            atomic_operators: The operators of which to compute the product when 
                evaluating the operator expression.
        """
        self._operators = tuple(atomic_operators)
        if len(self._operators) == 0:
            self._operators = (ScalarOperator.const(1),)
        super().__init__((self,))

    def _evaluate(self: ProductOperator,
                  arithmetics: OperatorArithmetics[TEval]) -> TEval:
        """
        Helper function used for evaluating operator expressions and computing arbitrary values
        during evaluation. The value to be computed is defined by the OperatorArithmetics.
        The evaluation guarantees that addition and multiplication of two operators will only
        be called when both operators act on the same degrees of freedom, and the tensor product
        will only be computed if they act on different degrees of freedom. 
        """

        def padded_op(op: ElementaryOperator | ScalarOperator,
                      degrees: Iterable[int]):
            # Creating the tensor product with op being last is most efficient.
            def accumulate_ops() -> Generator[TEval]:
                for degree in degrees:
                    if not degree in op._degrees:
                        yield ElementaryOperator.identity(degree)._evaluate(
                            arithmetics)
                yield op._evaluate(arithmetics)

            evaluated_ops = accumulate_ops()
            padded = next(evaluated_ops)
            for value in evaluated_ops:
                padded = arithmetics.tensor(padded, value)
            return padded

        # Sorting the degrees to avoid unnecessary permutations during the padding.
        noncanon_degrees = frozenset(
            (degree for op in self._operators for degree in op._degrees))
        degrees = _OperatorHelpers.canonicalize_degrees(noncanon_degrees)
        evaluated = padded_op(self._operators[0], degrees)
        for op in self._operators[1:]:
            if len(op._degrees) != 1 or op != ElementaryOperator.identity(
                    op._degrees[0]):
                evaluated = arithmetics.mul(evaluated, padded_op(op, degrees))
        return evaluated

    def __mul__(self: ProductOperator, other: Any) -> ProductOperator:
        if type(other) == ProductOperator:
            return ProductOperator((*self._operators, *other._operators))
        elif isinstance(other, (complex, float, int)):
            return ProductOperator(
                (*self._operators, ScalarOperator.const(other)))
        elif type(other) == ElementaryOperator or type(other) == ScalarOperator:
            return ProductOperator((*self._operators, other))
        return NotImplemented

    def __truediv__(self: ProductOperator, other: Any) -> ProductOperator:
        if isinstance(other, (complex, float, int)):
            other_op = ScalarOperator.const(1 / other)
            return ProductOperator((other_op, *self._operators))
        if type(other) == ScalarOperator:
            return ProductOperator((1 / other, *self._operators))
        return NotImplemented

    def __add__(self: ProductOperator, other: Any) -> OperatorSum:
        if type(other) == ProductOperator:
            return OperatorSum((self, other))
        elif isinstance(other, OperatorSum) or isinstance(
                other, (complex, float, int)):
            return OperatorSum((self,)) + other
        return NotImplemented

    def __sub__(self: ProductOperator, other: Any) -> OperatorSum:
        return self + (-1 * other)

    def __rmul__(self: ProductOperator, other: Any) -> ProductOperator:
        if isinstance(other, (complex, float, int)):
            return ProductOperator(
                (ScalarOperator.const(other), *self._operators))
        elif type(other) == ScalarOperator or type(other) == ElementaryOperator:
            return ProductOperator((other, *self._operators))
        return NotImplemented

    def __radd__(self: ProductOperator, other: Any) -> OperatorSum:
        return self + other  # Operator addition is commutative.

    def __rsub__(self: ProductOperator, other: Any) -> OperatorSum:
        minus_self = ProductOperator(
            (ScalarOperator.const(-1), *self._operators))
        return minus_self + other  # Operator addition is commutative.


class ElementaryOperator(ProductOperator):
    """
    Represents an elementary operator that acts on one or more degrees of freedom, 
    and cannot be further simplified to be the product or sum of other operators. 
    An elementary operator is defined as a function of zero or more complex-valued 
    parameters using the :func:`~dynamics.ElementaryOperator.define` class method.

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

        __slots__ = ['_id', '_parameters', '_expected_dimensions', '_generator']

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
                self: ElementaryOperator.Definition) -> tuple[int]:
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
    def define(cls, op_id: str, expected_dimensions: Sequence[int],
               create: Callable[..., NDArray[numpy.complexfloating]]) -> None:
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
        """
        if op_id in cls._ops:
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

    def __eq__(self: OperatorSum, other: Any) -> bool:
        """
        Returns:
            True, if the other value is an elementary operator with the same id
            acting on the same degrees of freedom, and False otherwise.
        """
        return type(
            other
        ) == self.__class__ and self._id == other._id and self._degrees == other._degrees

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
    def expected_dimensions(self: ElementaryOperator) -> tuple[int]:
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
                  arithmetics: OperatorArithmetics[TEval]) -> TEval:
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
        return self + (-1 * other)

    def __rmul__(self: ElementaryOperator, other: Any) -> ProductOperator:
        return other * ProductOperator((self,))

    def __radd__(self: ElementaryOperator, other: Any) -> OperatorSum:
        return self + other  # Operator addition is commutative.

    def __rsub__(self: ElementaryOperator, other: Any) -> OperatorSum:
        minus_self = ProductOperator((ScalarOperator.const(-1), self))
        return minus_self + other  # Operator addition is commutative.


class ScalarOperator(ProductOperator):
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

        __slots__ = ['_parameters', '_generator']

        def __init__(self: ScalarOperator.Definition,
                     generator: Callable[..., NumericType],
                     parameter_info: Optional[Callable[[], Mapping[str, str]]],
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
            # A variable number of arguments (i.e. `*args`) cannot be supported
            # for generators; it would prevent proper argument handling while
            # supporting additions and multiplication of all kinds of operators.
            arg_spec = inspect.getfullargspec(generator)
            if arg_spec.varargs is not None:
                raise ValueError(
                    f"generator for a '{type(self).__name__}' must not take *args"
                )
            if parameter_info is None:
                parameters = {}
                for arg_name in arg_spec.args + arg_spec.kwonlyargs:
                    parameters[arg_name] = _OperatorHelpers.parameter_docs(
                        arg_name, generator.__doc__)
                parameter_info = lambda: parameters
            # We need a function to retrieve information about what parameters
            # are required to invoke the generator, to ensure that the information
            # accurately captures any updates to the generators of sub-operators.
            self._parameters = parameter_info
            self._generator = generator

        @property
        def parameters(
                self: ScalarOperator.Definition
        ) -> Callable[[], Mapping[str, str]]:
            """
            Contains information about the parameters required to evaluate the
            generator function of the operator. The keys are the parameter names, 
            and the values are their documentation (if available).
            """
            return self._parameters

        @property
        def generator(
                self: ScalarOperator.Definition) -> Callable[..., NumericType]:
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
        instance = cls(lambda: constant_value)
        instance._constant_value = constant_value
        return instance

    __slots__ = ['_degrees', '_definition', '_constant_value']

    def __init__(
        self: ScalarOperator,
        generator: Callable[..., NumericType],
        parameter_info: Optional[Callable[[], Mapping[str,
                                                      str]]] = None) -> None:
        """
        Instantiates a scalar operator.

        Arguments:
            generator: The value of the scalar operator as a function of its
                parameters. The generator may take any number of complex-valued
                arguments and must return a number. Each parameter must be passed
                as a keyword argument when evaluating the operator. 
        """
        self._degrees: tuple[int] = ()
        self._definition = ScalarOperator.Definition(generator, parameter_info,
                                                     self._create_key)
        self._constant_value: Optional[NumericType] = None
        super().__init__((self,))

    def __eq__(self: ScalarOperator, other: Any) -> bool:
        """
        Returns:
            True, if the other value is a scalar operator with the same generator.
        """
        if type(other) != self.__class__:
            return False
        elif self._constant_value is None or other._constant_value is None:
            return self._definition.generator == other._definition.generator
        return self._constant_value == other._constant_value

    @property
    def generator(self: ScalarOperator) -> Callable[..., NumericType]:
        """
        The function that generates the value of the scalar operator. 
        The function can take any number of complex-valued arguments
        and returns a number.
        """
        return self._definition.generator

    @generator.setter
    def generator(self: ScalarOperator,
                  generator: Callable[..., NumericType]) -> None:
        """
        Sets the generator function. The generator function must not take a 
        variable number of arguments (that is it must not take `*args`), and 
        its parameter names should be descriptive keywords and the parameters 
        should be documented following Google documentation comment style. 

        Setting the generator of a scalar operator updates its evaluated value
        in all operators that contain the scalar operator.
        """
        self._definition = ScalarOperator.Definition(generator, None,
                                                     self._create_key)

    @property
    def parameters(self: ScalarOperator) -> Mapping[str, str]:
        """
        A mapping that contains the documentation comment for each parameter 
        needed to evaluate the generator.
        """
        return self._definition.parameters()

    @property
    def _is_spinop(self: ScalarOperator) -> bool:
        return True  # supported as coefficient

    def _invoke(self: ScalarOperator, **kwargs: NumericType) -> NumericType:
        """
        Helper function that extracts the necessary arguments from the given keyword 
        arguments and invokes the generator. 
        """
        generator_args, remaining_kwargs = _OperatorHelpers.args_from_kwargs(
            self.generator, **kwargs)
        evaluated = self.generator(*generator_args, **remaining_kwargs)
        if not isinstance(evaluated, (complex, float, int)):
            raise TypeError(
                f"generator of {type(self).__name__} must return a number")
        return evaluated

    def _evaluate(self: ScalarOperator,
                  arithmetics: OperatorArithmetics[TEval]) -> TEval:
        """
        Helper function for consistency with other operator expressions.
        Invokes the `evaluate` method of the given OperatorArithmetics.
        """
        return arithmetics.evaluate(self)

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
        if self._constant_value is not None:
            return str(self._constant_value)
        parameter_names = ", ".join(self.parameters)
        return f"{self.generator.__name__ or 'f'}({parameter_names})"

    def _compose(
        self: ScalarOperator, other: Any,
        fct: Callable[[NumericType, NumericType],
                      NumericType]) -> ScalarOperator:
        """
        Helper function to avoid duplicate code in the various arithmetic 
        operations supported on a ScalarOperator.
        """
        if isinstance(other, (complex, float, int)):
            if self._constant_value is None:
                generator = lambda **kwargs: fct(self._invoke(**kwargs), other)
                return ScalarOperator(generator, self._definition.parameters)
            return ScalarOperator.const(fct(self._constant_value, other))
        elif type(other) == ScalarOperator:
            if self._constant_value is None or other._constant_value is None:
                generator = lambda **kwargs: fct(self._invoke(**kwargs),
                                                 other._invoke(**kwargs))
                parameter_info = lambda: _OperatorHelpers.aggregate_parameters([
                    self._definition.parameters(),
                    other._definition.parameters()
                ])
                return ScalarOperator(generator, parameter_info)
            return ScalarOperator.const(
                fct(self._constant_value, other._constant_value))
        return NotImplemented

    def __pow__(self: ScalarOperator, other: Any) -> ScalarOperator:
        return self._compose(other, lambda v1, v2: v1**v2)

    def __mul__(self: ScalarOperator, other: Any) -> ScalarOperator:
        return self._compose(other, lambda v1, v2: v1 * v2)

    def __truediv__(self: ScalarOperator, other: Any) -> ScalarOperator:
        return self._compose(other, lambda v1, v2: v1 / v2)

    def __add__(self: ScalarOperator, other: Any) -> ScalarOperator:
        return self._compose(other, lambda v1, v2: v1 + v2)

    def __sub__(self: ScalarOperator, other: Any) -> ScalarOperator:
        return self._compose(other, lambda v1, v2: v1 - v2)

    def __rpow__(self: ScalarOperator, other: Any) -> ScalarOperator:
        return self._compose(other, lambda v1, v2: v2**v1)

    def __rmul__(self: ScalarOperator, other: Any) -> ScalarOperator:
        return self._compose(other, lambda v1, v2: v2 * v1)

    def __rtruediv__(self: ScalarOperator, other: Any) -> ScalarOperator:
        return self._compose(other, lambda v1, v2: v2 / v1)

    def __radd__(self: ScalarOperator, other: Any) -> ScalarOperator:
        return self._compose(other, lambda v1, v2: v2 + v1)

    def __rsub__(self: ScalarOperator, other: Any) -> ScalarOperator:
        return self._compose(other, lambda v1, v2: v2 - v1)


# Doc strings for type alias are not supported in Python.
# The string below hence merely serves to document it here;
# within the Python AST it is not associated with the type alias.
Operator = OperatorSum | ProductOperator | ElementaryOperator | ScalarOperator
"""
Type of an arbitrary operator expression. 
Operator expressions cannot be used within quantum kernels, but 
they provide methods to convert them to data types that can.
"""
