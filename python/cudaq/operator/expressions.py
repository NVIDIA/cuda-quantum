from __future__ import annotations
import inspect, numpy, operator # type: ignore
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Generator, Optional
from numpy.typing import NDArray

from .helpers import _OperatorHelpers, NumericType
from .manipulation import MatrixArithmetics, OperatorArithmetics, PrettyPrint
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime

class OperatorSum:
    """
    Represents an operator expression consisting of a sum of terms, 
    where each term is a product of elementary and scalar operators. 

    Operator expressions cannot be used within quantum kernels, but 
    they provide methods to convert them to data types that can.
    """

    __slots__ = ['_terms']
    def __init__(self: OperatorSum, terms: Sequence[ProductOperator] = []) -> None:
        """
        Instantiates an operator expression that represents a sum of terms,
        where each term is a product of elementary and scalar operators. 

        Arguments:
            terms: The ProductOperators that should be summed up when 
                evaluating the operator expression.
        """
        if len(terms) == 0:
            terms = [ProductOperator([ScalarOperator.const(0)])]
        self._terms = terms

    def __eq__(self: OperatorSum, other: Any) -> bool:
        """
        Returns:
            True, if the other value is an OperatorSum with equivalent terms, 
            and False otherwise. The equality takes into account that operator
            addition is commutative, as is the product of two operators if they
            act on different degrees of freedom.
            The equality comparison does *not* take commutation relations into 
            account, and does not try to reorder terms blockwise; it may hence 
            evaluate to False, even if two operators in reality are the same.
            If the equality evaluates to True, on the other hand, the operators 
            are guaranteed to represent the same transformation for all arguments.
        """
        if type(other) == self.__class__:
            def canonical_terms(prod: ProductOperator) -> Sequence[ElementaryOperator | ScalarOperator]:
                degrees = prod.degrees
                if len(degrees) == 1: return prod._operators
                elif len(degrees) != len(set(degrees)): 
                    # Some degrees exist multiple times; order the scalars, 
                    # but do not otherwise try to reorder terms.
                    scalars = [op for op in prod._operators if isinstance(op, ScalarOperator)]
                    non_scalars = [op for op in prod._operators if not isinstance(op, ScalarOperator)]
                    scalars.sort(key = lambda op: str(op._constant_value))
                    return [*scalars, *non_scalars]
                return sorted(prod._operators, key = lambda op: list(op._degrees))

            self_terms = (canonical_terms(term) for term in self._terms)
            other_terms = (canonical_terms(term) for term in other._terms)
            compare = lambda ops: "*".join([str(op) for op in ops])
            return sorted(self_terms, key = compare) == sorted(other_terms, key = compare)
        return False

    @property
    def degrees(self: OperatorSum) -> Sequence[int]:
        """
        The degrees of freedom that the operator acts on in canonical order.
        """
        degrees = list(set((degree for term in self._terms for op in term._operators for degree in op._degrees)))
        degrees.sort() # Sorted in canonical order to match the to_matrix method.
        return degrees

    @property
    def parameters(self: OperatorSum) -> Mapping[str, str]:
        """
        A mapping that contains the documentation comment for each parameter 
        needed to evaluate the operator expression.
        """
        return _OperatorHelpers.aggregate_parameters((op.parameters for term in self._terms for op in term._operators))

    def _evaluate(self: OperatorSum, arithmetics : OperatorArithmetics[TEval]) -> TEval:
        """
        Helper function used for evaluating operator expressions and computing arbitrary values
        during evaluation. The value to be computed is defined by the OperatorArithmetics.
        The evaluation guarantees that addition and multiplication of two operators will only
        be called when both operators act on the same degrees of freedom, and the tensor product
        will only be computed if they act on different degrees of freedom. 
        """
        degrees = set([degree for term in self._terms for op in term._operators for degree in op._degrees])
        # We need to make sure all matrices are of the same size to sum them up.
        def padded_term(term: ProductOperator) -> ProductOperator:
            for degree in degrees:
                if not degree in [op_degree for op in term._operators for op_degree in op._degrees]:
                    term *= ElementaryOperator.identity(degree)
            return term
        sum = padded_term(self._terms[0])._evaluate(arithmetics)
        for term in self._terms[1:]:
            sum = arithmetics.add(sum, padded_term(term)._evaluate(arithmetics))
        return sum

    def to_matrix(self: OperatorSum, dimensions: Mapping[int, int], **kwargs: NumericType) -> NDArray[numpy.complexfloating]:
        """
        Arguments:
            dimensions: A mapping that specifies the number of levels, that is
                the dimension, of each degree of freedom that the operator acts on.
            **kwargs: Keyword arguments needed to evaluate the operator. All
                required parameters and their documentation, if available, can be 
                queried by accessing the `parameter` property.

        Returns:
            The matrix representation of the operator expression in canonical order.
        """
        return self._evaluate(MatrixArithmetics(dimensions, **kwargs)).matrix

    def to_pauli_word(self: OperatorSum) -> cudaq_runtime.pauli_word:
        """
        Creates a representation of the operator as `pauli_word` that can be passed
        as an argument to quantum kernels.
        Raises a ValueError if the operator contains non-Pauli suboperators.
        """
        return self._evaluate(PauliWordConversion()).pauli_word

    def __str__(self: OperatorSum) -> str:
        return self._evaluate(PrettyPrint())

    def __mul__(self: OperatorSum, other: Any) -> OperatorSum:
        if type(other) == OperatorSum:
            return OperatorSum([self_term * other_term for self_term in self._terms for other_term in other._terms])
        elif isinstance(other, OperatorSum) or isinstance(other, (complex, float, int)):
            return OperatorSum([self_term * other for self_term in self._terms])
        return NotImplemented

    def __add__(self: OperatorSum, other: Any) -> OperatorSum:
        if type(other) == OperatorSum:
            return OperatorSum([*self._terms, *other._terms])
        elif isinstance(other, (complex, float, int)):
            other_term = ProductOperator([ScalarOperator.const(other)])
            return OperatorSum([*self._terms, other_term])
        elif type(other) == ScalarOperator: # Elementary and product operators are handled by their classes.
            return OperatorSum([*self._terms, ProductOperator([other])])
        return NotImplemented        

    def __sub__(self: OperatorSum, other: Any) -> OperatorSum:
        return self + (-1 * other)

    def __rmul__(self: OperatorSum, other: Any) -> OperatorSum:
        if isinstance(other, (complex, float, int)):
            return OperatorSum([self_term * other for self_term in self._terms])
        elif type(other) == ProductOperator:
            return OperatorSum([other]) * self
        elif type(other) == ScalarOperator or type(other) == ElementaryOperator:
            return OperatorSum([ProductOperator([other])]) * self
        return NotImplemented

    def __radd__(self: OperatorSum, other: Any) -> OperatorSum:
        if isinstance(other, (complex, float, int)):
            other_term = ProductOperator([ScalarOperator.const(other)])
            return OperatorSum([other_term, *self._terms])
        elif type(other) == ScalarOperator: # Elementary and product operators are handled by their classes.
            return OperatorSum([ProductOperator([other]), *self._terms])
        return NotImplemented

    def __rsub__(self: OperatorSum, other: Any) -> OperatorSum:
        return (-1 * self) + other # Operator addition is commutative.

class ProductOperator(OperatorSum):
    """
    Represents an operator expression consisting of a product of elementary
    and scalar operators. 

    Operator expressions cannot be used within quantum kernels, but 
    they provide methods to convert them to data types that can.
    """

    __slots__ = ['_operators']
    def __init__(self: ProductOperator, atomic_operators : Sequence[ElementaryOperator | ScalarOperator] = []) -> None:
        """
        Instantiates an operator expression that represents a product of elementary
        or scalar operators.

        Arguments:
            atomic_operators: The operators of which to compute the product when 
                evaluating the operator expression.
        """
        if len(atomic_operators) == 0:
            atomic_operators = [ScalarOperator.const(1)]
        self._operators = atomic_operators
        super().__init__([self])

    def _evaluate(self: ProductOperator, arithmetics : OperatorArithmetics[TEval]) -> TEval:
        """
        Helper function used for evaluating operator expressions and computing arbitrary values
        during evaluation. The value to be computed is defined by the OperatorArithmetics.
        The evaluation guarantees that addition and multiplication of two operators will only
        be called when both operators act on the same degrees of freedom, and the tensor product
        will only be computed if they act on different degrees of freedom. 
        """
        def padded_op(op: ElementaryOperator | ScalarOperator, degrees: Sequence[int]):
            # Creating the tensor product with op being last is most efficient.
            def accumulate_ops() -> Generator[TEval]:
                for degree in degrees:
                    if not degree in op._degrees:
                        yield ElementaryOperator.identity(degree)._evaluate(arithmetics)
                yield op._evaluate(arithmetics)
            evaluated_ops = accumulate_ops()
            padded = next(evaluated_ops)
            for value in evaluated_ops:
                padded = arithmetics.tensor(padded, value)
            return padded
        # Sorting the degrees to avoid unnecessary permutations during the padding.
        degrees = sorted(set([degree for op in self._operators for degree in op._degrees]))
        evaluated = padded_op(self._operators[0], degrees)
        for op in self._operators[1:]:
            if len(op._degrees) != 1 or op != ElementaryOperator.identity(op._degrees[0]):
                evaluated = arithmetics.mul(evaluated, padded_op(op, degrees))
        return evaluated

    def __mul__(self: ProductOperator, other: Any) -> ProductOperator:
        if type(other) == ProductOperator:
            return ProductOperator([*self._operators, *other._operators])
        elif isinstance(other, (complex, float, int)):
            return ProductOperator([*self._operators, ScalarOperator.const(other)])
        elif type(other) == ElementaryOperator or type(other) == ScalarOperator:
            return ProductOperator([*self._operators, other])
        return NotImplemented

    def __add__(self: ProductOperator, other: Any) -> OperatorSum:
        if type(other) == ProductOperator:
            return OperatorSum([self, other])
        elif isinstance(other, OperatorSum) or isinstance(other, (complex, float, int)):
            return OperatorSum([self]) + other
        return NotImplemented

    def __sub__(self: ProductOperator, other: Any) -> OperatorSum:
        return self + (-1 * other)

    def __rmul__(self: ProductOperator, other: Any) -> ProductOperator:
        if isinstance(other, (complex, float, int)):
            return ProductOperator([ScalarOperator.const(other), *self._operators])
        elif type(other) == ScalarOperator or type(other) == ElementaryOperator:
            return ProductOperator([other, *self._operators])
        return NotImplemented

    def __radd__(self: ProductOperator, other: Any) -> OperatorSum:
        return self + other # Operator addition is commutative.

    def __rsub__(self: ProductOperator, other: Any) -> OperatorSum:
        minus_self = ProductOperator([ScalarOperator.const(-1), *self._operators])
        return minus_self + other # Operator addition is commutative.

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
        def __init__(self: ElementaryOperator.Definition, 
                     id: str, 
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
                raise ValueError(f"an ElementaryOperator needs to act on at least one degree " +
                                  "of freedom - use a ScalarOperator to define operators that " +
                                  "do not act on any degrees of freedom")

            forwarded_as_kwarg = [["dimensions", "dims"], ["dimension", "dim"]]
            def with_dimension_check(creation: Callable, dimensions: Sequence[int], **kwargs) -> NDArray[numpy.complexfloating]:
                if any([expected > 0 and dimensions[i] != expected for i, expected in enumerate(expected_dimensions)]):
                    raise ValueError(f'no built-in operator {id} has been defined '\
                                    f'for {len(dimensions)} degree(s) of freedom with dimension(s) {dimensions}')
                # If the population of kwargs here is changed, adjust the filtering 
                # in the `parameters` property below.
                for forwarded in forwarded_as_kwarg[0]:
                    kwargs[forwarded] = kwargs.get(forwarded, dimensions) # add if it does not exist
                if len(dimensions) == 1: 
                    for forwarded in forwarded_as_kwarg[1]:
                        kwargs[forwarded] = kwargs.get(forwarded, dimensions[0]) # add if it does not exist
                creation_args, remaining_kwargs = _OperatorHelpers.args_from_kwargs(creation, **kwargs)
                evaluated = creation(*creation_args, **remaining_kwargs)
                if not isinstance(evaluated, numpy.ndarray):
                    raise TypeError("operator evaluation must return a NDArray[complex]")
                return evaluated

            parameters, forwarded = {}, [keyword for group in forwarded_as_kwarg for keyword in group]
            arg_spec = inspect.getfullargspec(create)
            for pname in arg_spec.args + arg_spec.kwonlyargs:
                if not pname in forwarded:
                    parameters[pname] = _OperatorHelpers.parameter_docs(pname, create.__doc__)

            self._id = id
            self._parameters = parameters
            self._expected_dimensions = expected_dimensions
            self._generator = lambda dimensions, **kwargs: with_dimension_check(create, dimensions, **kwargs)

        @property
        def id(self: ElementaryOperator.Definition) -> str:
            """
            A unique identifier of the operator.
            """
            return self._id

        @property
        def parameters(self: ElementaryOperator.Definition) -> Mapping[str, str]:
            """
            Contains information about the parameters required to generate the matrix
            representation of the operator. The keys are the parameter names, 
            and the values are their documentation (if available).
            """
            return self._parameters

        @property
        def expected_dimensions(self: ElementaryOperator.Definition) -> Sequence[int]:
            """
            The number of levels, that is the dimension, for each degree of freedom
            in canonical order that the operator acts on. A value of zero or less 
            indicates that the operator is defined for any dimension of that degree.
            """
            return self._expected_dimensions
        
        @property
        def generator(self: ElementaryOperator.Definition) -> Callable[..., NDArray[numpy.complexfloating]]:
            """
            A callable that takes any number of complex-valued keyword arguments and 
            returns the matrix representing the operator in canonical order.
            The parameter names and their documentation (if available) can be accessed
            via the `parameter` property.
            """
            return self._generator

    _ops : dict[str, ElementaryOperator.Definition] = {}
    """
    Contains the generator for each defined ElementaryOperator.
    The generator takes a dictionary defining the dimensions for each degree of freedom,
    as well as keyword arguments for complex values.
    """

    @classmethod
    def define(cls, op_id: str, expected_dimensions: Sequence[int], create: Callable[..., NDArray[numpy.complexfloating]]) -> None:
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
            raise ValueError(f"an ElementaryOperator with id {op_id} already exists")
        cls._ops[op_id] = cls.Definition(op_id, expected_dimensions, create, cls._create_key)

    @classmethod
    def zero(cls, degree: int) -> ElementaryOperator:
        op_id = "zero"
        if not op_id in cls._ops:
            cls.define(op_id, [0], 
                lambda dim: numpy.zeros((dim, dim), dtype=numpy.complex128))
        return cls(op_id, [degree])

    @classmethod
    def identity(cls, degree: int) -> ElementaryOperator:
        op_id = "identity"
        if not op_id in cls._ops:
            cls.define(op_id, [0], 
                lambda dim: numpy.diag(numpy.ones(dim, dtype=numpy.complex128)))
        return cls(op_id, [degree])

    __slots__ = ['_id', '_degrees']
    def __init__(self: ElementaryOperator, operator_id: str, degrees: Sequence[int]) -> None:
        """
        Instantiates an elementary operator.

        Arguments:
            operator_id: The id of the operator as specified when it was defined.
            degrees: The degrees of freedom that the operator acts on.
        """
        if not operator_id in ElementaryOperator._ops:
            raise ValueError(f"no built-in operator '{operator_id}' has been defined")
        self._id = operator_id
        self._degrees = sorted(degrees) # Sorting so that order matches the elementary operation definition.
        num_degrees = len(self.expected_dimensions)
        if len(degrees) != num_degrees:
            raise ValueError(f"definition of {operator_id} acts on {num_degrees} degree(s) of freedom (given: {len(degrees)})")
        super().__init__([self])

    def __eq__(self: OperatorSum, other: Any) -> bool:
        """
        Returns:
            True, if the other value is an elementary operator with the same id
            acting on the same degrees of freedom, and False otherwise.
        """
        if type(other) == self.__class__:
            attr_getters = [operator.attrgetter(attr) for attr in self.__slots__]
            return all(getter(self) == getter(other) for getter in attr_getters)
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
    def expected_dimensions(self: ElementaryOperator) -> Sequence[int]:
        """
        The number of levels, that is the dimension, for each degree of freedom
        in canonical order that the operator acts on. A value of zero or less 
        indicates that the operator is defined for any dimension of that degree.
        """
        return ElementaryOperator._ops[self._id].expected_dimensions

    def _evaluate(self: ElementaryOperator, arithmetics: OperatorArithmetics[TEval]) -> TEval:
        """
        Helper function for consistency with other operator expressions.
        Invokes the `evaluate` method of the given OperatorArithmetics.
        """
        return arithmetics.evaluate(self)

    def to_matrix(self: ElementaryOperator, dimensions: Mapping[int, int], **kwargs: NumericType) -> NDArray[numpy.complexfloating]:
        """
        Arguments:
            dimensions: A mapping that specifies the number of levels, that is
                the dimension, of each degree of freedom that the operator acts on.
            **kwargs: Keyword arguments needed to evaluate the operator. All
                required parameters and their documentation, if available, can be 
                queried by accessing the `parameter` property.

        Returns:
            The matrix representation of the operator expression in canonical order.
        """
        missing_degrees = [degree not in dimensions for degree in self._degrees]
        if any(missing_degrees):
            raise ValueError(f'missing dimensions for degree(s) {[self._degrees[i] for i, x in enumerate(missing_degrees) if x]}')
        relevant_dimensions = [dimensions[d] for d in self._degrees]
        return ElementaryOperator._ops[self._id].generator(relevant_dimensions, **kwargs)

    def __str__(self: ElementaryOperator) -> str:
        parameter_names = ", ".join(self.parameters)
        if parameter_names != "": parameter_names = f"({parameter_names})"
        return f"{self._id}{parameter_names}{self._degrees}"

    def __mul__(self: ElementaryOperator, other: Any) -> ProductOperator:
        if type(other) == ElementaryOperator:
            return ProductOperator([self, other])
        elif isinstance(other, OperatorSum) or isinstance(other, (complex, float, int)):
            return ProductOperator([self]) * other
        return NotImplemented

    def __add__(self: ElementaryOperator, other: Any) -> OperatorSum:
        if type(other) == ElementaryOperator:
            op1 = ProductOperator([self])
            op2 = ProductOperator([other])
            return OperatorSum([op1, op2])
        elif isinstance(other, OperatorSum) or isinstance(other, (complex, float, int)):
            return OperatorSum([ProductOperator([self])]) + other
        return NotImplemented

    def __sub__(self: ElementaryOperator, other: Any) -> OperatorSum:
        return self + (-1 * other)

    def __rmul__(self: ElementaryOperator, other: Any) -> ProductOperator:
        return other * ProductOperator([self])

    def __radd__(self: ElementaryOperator, other: Any) -> OperatorSum:
        return self + other # Operator addition is commutative.

    def __rsub__(self: ElementaryOperator, other: Any) -> OperatorSum:
        minus_self = ProductOperator([ScalarOperator.const(-1), self])
        return minus_self + other # Operator addition is commutative.

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
                   "operator definitions must be created using the `ScalarOperator.generator` setter"
            # A variable number of arguments (i.e. *args) cannot be supported
            # for generators; it would prevent proper argument handling while 
            # supporting additions and multiplication of all kinds of operators.
            arg_spec = inspect.getfullargspec(generator)
            if arg_spec.varargs is not None:
                raise ValueError(f"generator for a '{type(self).__name__}' must not take *args")
            if parameter_info is None:
                parameters = {}
                for arg_name in arg_spec.args + arg_spec.kwonlyargs:
                    parameters[arg_name] = _OperatorHelpers.parameter_docs(arg_name, generator.__doc__)
                parameter_info = lambda: parameters
            # We need a function to retrieve information about what parameters 
            # are required to invoke the generator, to ensure that the information 
            # accurately captures any updates to the generators of sub-operators.    
            self._parameters = parameter_info
            self._generator = generator

        @property
        def parameters(self: ScalarOperator.Definition) -> Callable[[], Mapping[str, str]]:
            """
            Contains information about the parameters required to evaluate the
            generator function of the operator. The keys are the parameter names, 
            and the values are their documentation (if available).
            """
            return self._parameters
        
        @property
        def generator(self: ScalarOperator.Definition) -> Callable[..., NumericType]:
            """
            A callable that takes any number of complex-valued keyword arguments and 
            returns the scalar value representing the evaluated operator.
            The parameter names and their documentation (if available) can be accessed
            via the `parameter` property.
            """
            return self._generator


    @classmethod
    def const(cls, constant_value: NumericType) -> ScalarOperator:
        instance = cls(lambda: constant_value)
        instance._constant_value = constant_value
        return instance

    __slots__ = ['_degrees', '_definition', '_constant_value']
    def __init__(self: ScalarOperator, generator: Callable[..., NumericType], parameter_info: Optional[Callable[[], Mapping[str, str]]] = None) -> None:
        """
        Instantiates a scalar operator.

        Arguments:
            generator: The value of the scalar operator as a function of its
                parameters. The generator may take any number of complex-valued
                arguments and must return a number. Each parameter must be passed
                as a keyword argument when evaluating the operator. 
        """
        self._degrees : Sequence[int] = []
        self._definition = ScalarOperator.Definition(generator, parameter_info, self._create_key)
        self._constant_value : Optional[NumericType] = None
        super().__init__([self])

    def __eq__(self: ScalarOperator, other: Any) -> bool:
        """
        Returns:
            True, if the other value is a scalar operator with the same generator.
        """
        if type(other) != self.__class__: return False
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
    def generator(self: ScalarOperator, generator: Callable[..., NumericType]) -> None:
        """
        Sets the generator function. The generator function must not take a 
        variable number of arguments (that is it must not take `*args`), and 
        its parameter names should be descriptive keywords and the parameters 
        should be documented following Google documentation comment style. 

        Setting the generator of a scalar operator updates its evaluated value
        in all operators that contain the scalar operator.
        """
        self._definition = ScalarOperator.Definition(generator, None, self._create_key)

    @property
    def parameters(self: ScalarOperator) -> Mapping[str, str]:
        """
        A mapping that contains the documentation comment for each parameter 
        needed to evaluate the generator.
        """
        return self._definition.parameters()

    def _invoke(self: ScalarOperator, **kwargs: NumericType) -> NumericType:
        """
        Helper function that extracts the necessary arguments from the given keyword 
        arguments and invokes the generator. 
        """
        generator_args, remaining_kwargs = _OperatorHelpers.args_from_kwargs(self.generator, **kwargs)
        evaluated = self.generator(*generator_args, **remaining_kwargs)
        if not isinstance(evaluated, (complex, float, int)):
            raise TypeError(f"generator of {type(self).__name__} must return a number")
        return evaluated

    def _evaluate(self: ScalarOperator, arithmetics: OperatorArithmetics[TEval]) -> TEval:
        """
        Helper function for consistency with other operator expressions.
        Invokes the `evaluate` method of the given OperatorArithmetics.
        """
        return arithmetics.evaluate(self)

    # The argument `dimensions` here is only passed for consistency with parent classes.
    def to_matrix(self: ScalarOperator, dimensions: Mapping[int, int] = {}, **kwargs: NumericType) -> NDArray[numpy.complexfloating]:
        """
        Class method for consistency with other operator classes.
        Invokes the generator with the given keyword arguments.

        Arguments:
            dimensions: (unused, passed for consistency) 
                A mapping that specifies the number of levels, that is
                the dimension, of each degree of freedom that the operator acts on.
            **kwargs: Keyword arguments needed to evaluate the generator. All
                required parameters and their documentation, if available, can be 
                queried by accessing the `parameter` property.

        Returns:
            The scalar value of the operator for the given keyword arguments.
        """
        return numpy.array([self._invoke(**kwargs)], dtype=numpy.complex128)

    def __str__(self: ScalarOperator) -> str:
        if self._constant_value is not None:
            return str(self._constant_value)
        parameter_names = ", ".join(self.parameters)
        return f"{self.generator.__name__ or 'f'}({parameter_names})"

    def _compose(self: ScalarOperator, 
                 fct: Callable[[NumericType], NumericType], 
                 get_params: Optional[Callable[[], Mapping[str, str]]]) -> ScalarOperator:
        """
        Helper function to avoid duplicate code in the various arithmetic 
        operations supported on a ScalarOperator.
        """
        generator = lambda **kwargs: fct(self._invoke(**kwargs), **kwargs)
        if get_params is None:
            parameter_info = self._definition.parameters
        else:
            parameter_info = lambda: _OperatorHelpers.aggregate_parameters([self._definition.parameters(), get_params()])
        return ScalarOperator(generator, parameter_info)

    def __pow__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if type(other) == ScalarOperator:
            fct = lambda value, **kwargs: value ** other._invoke(**kwargs)
            return self._compose(fct, other._definition.parameters)
        elif isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: value ** other
            return self._compose(fct, None)
        return NotImplemented

    def __mul__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if type(other) == ScalarOperator:
            fct = lambda value, **kwargs: value * other._invoke(**kwargs)
            return self._compose(fct, other._definition.parameters)
        elif isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: value * other
            return self._compose(fct, None)
        return NotImplemented
    
    def __truediv__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if type(other) == ScalarOperator:
            fct = lambda value, **kwargs: value / other._invoke(**kwargs)
            return self._compose(fct, other._definition.parameters)
        elif isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: value / other
            return self._compose(fct, None)
        return NotImplemented

    def __add__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if type(other) == ScalarOperator:
            fct = lambda value, **kwargs: value + other._invoke(**kwargs)
            return self._compose(fct, other._definition.parameters)
        elif isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: value + other
            return self._compose(fct, None)
        return NotImplemented

    def __sub__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if type(other) == ScalarOperator:
            fct = lambda value, **kwargs: value - other._invoke(**kwargs)
            return self._compose(fct, other._definition.parameters)
        elif isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: value - other
            return self._compose(fct, None)
        return NotImplemented

    def __rpow__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: other ** value
            return self._compose(fct, None)
        return NotImplemented

    def __rmul__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: other * value
            return self._compose(fct, None)
        return NotImplemented

    def __rtruediv__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: other / value
            return self._compose(fct, None)
        return NotImplemented

    def __radd__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: other + value
            return self._compose(fct, None)
        return NotImplemented

    def __rsub__(self: ScalarOperator, other: Any) -> ScalarOperator:
        if isinstance(other, (complex, float, int)):
            fct = lambda value, **kwargs: other - value
            return self._compose(fct, None)
        return NotImplemented

# Doc strings for type alias are not supported in Python.
# The string below hence merely serves to document it here;
# within the Python AST it is not associated with the type alias.
Operator = OperatorSum | ProductOperator | ElementaryOperator | ScalarOperator
"""
Type of an arbitrary operator expression. 
Operator expressions cannot be used within quantum kernels, but 
they provide methods to convert them to data types that can.
"""
