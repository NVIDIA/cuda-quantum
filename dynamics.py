from __future__ import annotations
import cudaq, inspect, itertools, numpy, os, operator, re, scipy, sys # type: ignore
from abc import ABC, abstractmethod
from collections.abc import Mapping, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Generic, Generator, Iterable, Optional, SupportsComplex, TypeVar
from numpy.typing import NDArray

if (3, 11) <= sys.version_info: NumericType = SupportsComplex
else: NumericType = numpy.complexfloating | complex | float | int

class _OperatorHelpers:

    @staticmethod
    def aggregate_parameters(parameter_mappings: Iterable[Mapping[str, str]]) -> Mapping[str, str]:
        """
        Helper function used by all operator classes to return a mapping with the
        used parameters and their respective description as defined in a doc comment.
        """
        param_descriptions : dict[str,str] = {}
        for descriptions in parameter_mappings:
            for key in descriptions:
                existing_desc = param_descriptions.get(key) or ""
                new_desc = descriptions[key]
                has_existing_desc = existing_desc is not None and existing_desc != ""
                has_new_desc = new_desc != ""
                if has_existing_desc and has_new_desc:
                    param_descriptions[key] = existing_desc + f'{os.linesep}---{os.linesep}' + new_desc
                elif has_existing_desc:
                    param_descriptions[key] = existing_desc
                else: 
                    param_descriptions[key] = new_desc
        return param_descriptions
    
    @staticmethod
    def parameter_docs(param_name: str, docs: Optional[str]) -> str:
        """
        Given the function documentation, tries to find the documentation for a 
        specific parameter. Expects Google docs style to be used.

        Returns:
            The documentation or an empty string if it failed to find it.
        """
        if param_name is None or docs is None:
            return ""
        
        # Using re.compile on the pattern before passing it to search
        # seems to behave slightly differently than passing the string.
        # Also, Python caches already used patterns, so compiling on
        # the fly seems fine.
        def keyword_pattern(word):
            return r"(?:^\s*" + word + ":\s*\r?\n)"
        def param_pattern(param_name): 
            return r"(?:^\s*" + param_name + r"\s*(\(.*\))?:)\s*(.*)$"

        param_docs = ""
        try: # Make sure failing to retrieve docs never cases an error.
            split = re.split(keyword_pattern("Arguments"), docs, flags=re.MULTILINE)
            if len(split) == 1:
                split = re.split(keyword_pattern("Args"), docs, flags=re.MULTILINE)
            if len(split) > 1:
                match = re.search(param_pattern(param_name), split[1], re.MULTILINE)
                if match is not None:
                    param_docs = match.group(2) + split[1][match.end(2):]
                    match = re.search(param_pattern("\S*?"), param_docs, re.MULTILINE)
                    if match is not None:
                        param_docs = param_docs[:match.start(0)]
                    param_docs = re.sub(r'\s+', ' ', param_docs)
        except: pass
        return param_docs.strip()

    @staticmethod
    def args_from_kwargs(fct: Callable, **kwargs: Any):
        """
        Extracts the positional argument and keyword only arguments 
        for the given function from the passed kwargs. 
        """
        arg_spec = inspect.getfullargspec(fct)
        signature = inspect.signature(fct)
        if arg_spec.varargs is not None:
            raise ValueError("cannot extract arguments for a function with a *args argument")
        consumes_kwargs = arg_spec.varkw is not None
        if consumes_kwargs:
            kwargs = kwargs.copy() # We will modify and return a copy

        def find_in_kwargs(arg_name: str):
            # Try to get the argument from the kwargs passed during operator 
            # evaluation.
            arg_value = kwargs.get(arg_name)
            if arg_value is None:
                # If no suitable keyword argument was defined, check if the 
                # generator defines a default value for this argument.
                default_value = signature.parameters[arg_name].default
                if default_value is not inspect.Parameter.empty:
                    arg_value = default_value
            elif consumes_kwargs:
                del kwargs[arg_name]
            if arg_value is None:
                raise ValueError(f'missing keyword argument {arg_name}')
            return arg_value

        extracted_args = []
        for arg_name in arg_spec.args:
            extracted_args.append(find_in_kwargs(arg_name))
        if consumes_kwargs:
            return extracted_args, kwargs
        elif len(arg_spec.kwonlyargs) > 0:
            # If we can't pass all remaining kwargs, 
            # we need to create a separate dictionary for kwonlyargs.
            kwonlyargs = {}
            for arg_name in arg_spec.kwonlyargs:
                kwonlyargs[arg_name] = find_in_kwargs(arg_name)
            return extracted_args, kwonlyargs
        return extracted_args, {}

    @staticmethod
    def generate_all_states(degrees: Sequence[int], dimensions: Mapping[int, int]):
        """
        Generates all possible states for the given dimensions ordered according to 
        the sequence of degrees (ordering is relevant if dimensions differ).
        """
        if len(degrees) == 0:
            return []
        states = [[str(state)] for state in range(dimensions[degrees[0]])]
        for d in degrees[1:]:
            prod = itertools.product(states, [str(state) for state in range(dimensions[d])])
            states = [current + [new] for current, new in prod]
        return [''.join(state) for state in states]

    @staticmethod
    def permute_matrix(matrix: NDArray[numpy.complexfloating], permutation: Sequence[int]) -> None:
        """
        Permutes the given matrix according to the given permutation.
        If states is the current order of vector entries on which the given matrix
        acts, and permuted_states is the desired order of an array on which the
        permuted matrix should act, then the permutation is defined such that
        [states[i] for i in permutation] produces permuted_states.
        """
        for i in range(numpy.size(matrix, 1)):
            matrix[:,i] = matrix[permutation,i]
        for i in range(numpy.size(matrix, 0)):
            matrix[i,:] = matrix[i,permutation]
 

TEval = TypeVar('TEval')

class OperatorArithmetics(ABC, Generic[TEval]):
    """
    This class serves as a monad base class for computing arbitrary values
    during operator evaluation.
    """

    @abstractmethod
    def evaluate(self: OperatorArithmetics[TEval], op: ElementaryOperator | ScalarOperator) -> TEval:
        """
        Accesses the relevant data to evaluate an operator expression in the leaf 
        nodes, that is in elementary and scalar operators.
        """
        pass

    @abstractmethod
    def add(self: OperatorArithmetics[TEval], val1: TEval, val2: TEval) -> TEval: 
        """
        Adds two operators that act on the same degrees of freedom.
        """
        pass

    @abstractmethod
    def mul(self: OperatorArithmetics[TEval], val1: TEval, val2: TEval) -> TEval: 
        """
        Multiplies two operators that act on the same degrees of freedom.
        """
        pass

    @abstractmethod
    def tensor(self: OperatorArithmetics[TEval], val1: TEval, val2: TEval) -> TEval: 
        """
        Computes the tensor product of two operators that act on different 
        degrees of freedom.
        """
        pass

class MatrixArithmetics(OperatorArithmetics['MatrixArithmetics.Evaluated']):
    """
    Encapsulates the functions needed to compute the matrix representation
    of an operator expression.
    """

    class Evaluated:
        """
        Stores the relevant data to compute the matrix representation of an
        operator expression.
        """

        def __init__(self: MatrixArithmetics.Evaluated, degrees: Sequence[int], matrix: NDArray[numpy.complexfloating]) -> None:
            """
            Instantiates an object that contains the matrix representation of an
            operator acting on the given degrees of freedom.

            Arguments:
                degrees: The degrees of freedom that the matrix applies to.
                matrix: The matrix representation of an evaluated operator.
            """
            self._degrees = degrees
            self._matrix = matrix

        @property
        def degrees(self: MatrixArithmetics.Evaluated) -> Sequence[int]: 
            """
            The degrees of freedom that the matrix of the evaluated value applies to.
            """
            return self._degrees

        @property
        def matrix(self: MatrixArithmetics.Evaluated) -> NDArray[numpy.complexfloating]: 
            """
            The matrix representation of an evaluated operator, ordered according
            to the sequence of degrees of freedom associated with the evaluated value.
            """
            return self._matrix

    def _canonicalize(self: MatrixArithmetics, op_matrix: NDArray[numpy.complexfloating], op_degrees: Sequence[int]) -> tuple[NDArray[numpy.complexfloating], Sequence[int]]:
        """
        Given a matrix representation that acts on the given degrees or freedom, 
        sorts the degrees and permutes the matrix to match that canonical order.

        Returns:
            A tuple consisting of the permuted matrix as well as the sequence of degrees
            of freedom in canonical order.
        """
        # FIXME: check endianness ... (in the sorting/states here, and in the matrix definitions)
        canon_degrees = sorted(op_degrees)
        if op_degrees != canon_degrees:
            # There may be a more efficient way, but I needed something correct first.
            states = _OperatorHelpers.generate_all_states(canon_degrees, self._dimensions)
            indices = dict([(d, idx) for idx, d in enumerate(canon_degrees)])
            reordering = [indices[op_degree] for op_degree in op_degrees]
            # [degrees[i] for i in reordering] produces op_degrees
            op_states = [''.join([state[i] for i in reordering]) for state in states]
            state_indices = dict([(state, idx) for idx, state in enumerate(states)])
            permutation = [state_indices[op_state] for op_state in op_states]
            # [states[i] for i in permutation] produces op_states
            _OperatorHelpers.permute_matrix(op_matrix, permutation)
            return op_matrix, canon_degrees
        return op_matrix, canon_degrees

    def tensor(self: MatrixArithmetics, op1: MatrixArithmetics.Evaluated, op2: MatrixArithmetics.Evaluated) -> MatrixArithmetics.Evaluated:
        """
        Computes the tensor product of two evaluate operators that act on different 
        degrees of freedom using `numpy.kron`.
        """
        assert len(set(op1.degrees).intersection(op2.degrees)) == 0, \
            "Operators should not have common degrees of freedom."
        op_degrees = [*op1.degrees, *op2.degrees]
        op_matrix = numpy.kron(op1.matrix, op2.matrix)
        new_matrix, new_degrees = self._canonicalize(op_matrix, op_degrees)
        return MatrixArithmetics.Evaluated(new_degrees, new_matrix)

    def mul(self: MatrixArithmetics, op1: MatrixArithmetics.Evaluated, op2: MatrixArithmetics.Evaluated) -> MatrixArithmetics.Evaluated:
        """
        Multiplies two evaluated operators that act on the same degrees of freedom
        using `numpy.dot`.
        """
        # Elementary operators have sorted degrees such that we have a unique convention 
        # for how to define the matrix. Tensor products permute the computed matrix if 
        # necessary to guarantee that all operators always have sorted degrees.
        assert op1.degrees == op2.degrees, "Operators should have the same order of degrees."
        return MatrixArithmetics.Evaluated(op1.degrees, numpy.dot(op1.matrix, op2.matrix))

    def add(self: MatrixArithmetics, op1: MatrixArithmetics.Evaluated, op2: MatrixArithmetics.Evaluated) -> MatrixArithmetics.Evaluated:
        """
        Multiplies two evaluated operators that act on the same degrees of freedom
        using `numpy`'s array addition.
        """
        # Elementary operators have sorted degrees such that we have a unique convention 
        # for how to define the matrix. Tensor products permute the computed matrix if 
        # necessary to guarantee that all operators always have sorted degrees.
        assert op1.degrees == op2.degrees, "Operators should have the same order of degrees."
        return MatrixArithmetics.Evaluated(op1.degrees, op1.matrix + op2.matrix)

    def evaluate(self: MatrixArithmetics, op: ElementaryOperator | ScalarOperator) -> MatrixArithmetics.Evaluated: 
        """
        Computes the matrix of an ElementaryOperator or ScalarOperator using its 
        `to_matrix` method.
        """
        matrix = op.to_matrix(self._dimensions, **self._kwargs)
        return MatrixArithmetics.Evaluated(op._degrees, matrix)

    def __init__(self: MatrixArithmetics, dimensions: Mapping[int, int], **kwargs: NumericType) -> None:
        """
        Instantiates a MatrixArithmetics instance that can act on the given
        dimensions.

        Arguments:
            dimensions: A mapping that specifies the number of levels, that 
                is the dimension, of each degree of freedom that the evaluated 
                operator can act on.
            **kwargs: Keyword arguments needed to evaluate, that is access data in,
                the leaf nodes of the operator expression. Leaf nodes are 
                values of type ElementaryOperator or ScalarOperator.
        """
        self._dimensions = dimensions
        self._kwargs = kwargs

class PrettyPrint(OperatorArithmetics[str]):

    def tensor(self, op1: str, op2: str) -> str:
        def add_parens(str_value: str):
            outer_str = re.sub(r'\(.+?\)', '', str_value)
            if " + " in outer_str or " * " in outer_str: return f"({str_value})"
            else: return str_value
        return f"{add_parens(op1)} x {add_parens(op2)}"

    def mul(self, op1: str, op2: str) -> str:
        def add_parens(str_value: str):
            outer_str = re.sub(r'\(.+?\)', '', str_value)
            if " + " in outer_str or " x " in outer_str: return f"({str_value})"
            else: return str_value
        return f"{add_parens(op1)} * {add_parens(op2)}"

    def add(self, op1: str, op2: str) -> str:
        return f"{op1} + {op2}"

    def evaluate(self, op: ElementaryOperator | ScalarOperator) -> str: 
        return str(op)

class PauliWordConversion(OperatorArithmetics[cudaq.pauli_word]):

    class Evaluated:
        """
        Stores the relevant data to compute the representation of an
        operator expression as a `pauli_word`.
        """

        def __init__(self: PauliWordConversion.Evaluated, degrees: Sequence[int], pauli_word: cudaq.pauli_word) -> None:
            """
            Instantiates an object that contains the `pauli_word` representation of an
            operator acting on the given degrees of freedom.

            Arguments:
                degrees: The degrees of freedom that the matrix applies to.
                pauli_word: The `pauli_word` representation of an evaluated operator.
            """
            self._degrees = degrees
            self._pauli_word = pauli_word

        @property
        def degrees(self: PauliWordConversion.Evaluated) -> Sequence[int]: 
            """
            The degrees of freedom that the evaluated operator acts on.
            """
            return self._degrees

        @property
        def pauli_word(self: PauliWordConversion.Evaluated) -> cudaq.pauli_word:
            """
            The `pauli_word` representation of an evaluated operator, ordered according
            to the sequence of degrees of freedom associated with the evaluated value.
            """
            return self._pauli_word

    def tensor(self, op1: PauliWordConversion.Evaluated, op2: PauliWordConversion.Evaluated) -> PauliWordConversion.Evaluated:
        raise NotImplementedError()

    def mul(self, op1: PauliWordConversion.Evaluated, op2: PauliWordConversion.Evaluated) -> PauliWordConversion.Evaluated:
        raise NotImplementedError()

    def add(self, op1: PauliWordConversion.Evaluated, op2: PauliWordConversion.Evaluated) -> PauliWordConversion.Evaluated:
        raise NotImplementedError()

    def evaluate(self, op: ElementaryOperator | ScalarOperator) -> PauliWordConversion.Evaluated:
        raise NotImplementedError()


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
        return [degree for term in self._terms for op in term._operators for degree in op._degrees]

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

    def to_pauli_word(self: OperatorSum) -> cudaq.pauli_word:
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

# Operators as defined here (watch out of differences in convention): 
# https://www.dynamiqs.org/python_api/utils/operators/sigmay.html
class operators:

    class matrices:
        @staticmethod
        def _create(dimension: int) -> NDArray[numpy.complexfloating]: 
            return numpy.diag(numpy.sqrt(numpy.arange(1, dimension, dtype=numpy.complex128)), -1)
        @staticmethod
        def _annihilate(dimension: int) -> NDArray[numpy.complexfloating]: 
            return numpy.diag(numpy.sqrt(numpy.arange(1, dimension, dtype=numpy.complex128)), 1)
        @staticmethod
        def _position(dimension: int) -> NDArray[numpy.complexfloating]:
            return complex(0.5) * (operators.matrices._create(dimension) + operators.matrices._annihilate(dimension))
        @staticmethod
        def _momentum(dimension: int) -> NDArray[numpy.complexfloating]:
            return 0.5j * (operators.matrices._create(dimension) - operators.matrices._annihilate(dimension))
        @staticmethod
        def _displace(dimension: int, displacement: NumericType) -> NDArray[numpy.complexfloating]:
            """Connects to the next available port.
            Args:
                displacement: Amplitude of the displacement operator.
                    See also https://en.wikipedia.org/wiki/Displacement_operator.
            """
            displacement = complex(displacement)
            term1 = displacement * operators.matrices._create(dimension)
            term2 = numpy.conjugate(displacement) * operators.matrices._annihilate(dimension)
            return scipy.linalg.expm(term1 - term2)
        @staticmethod
        def _squeeze(dimension: int, squeezing: NumericType) -> NDArray[numpy.complexfloating]:
            """Connects to the next available port.
            Args:
                squeezing: Amplitude of the squeezing operator.
                    See also https://en.wikipedia.org/wiki/Squeeze_operator.
            """
            squeezing = complex(squeezing)
            term1 = numpy.conjugate(squeezing) * numpy.linalg.matrix_power(operators.matrices._annihilate(dimension), 2)
            term2 = squeezing * numpy.linalg.matrix_power(operators.matrices._create(dimension), 2)
            return scipy.linalg.expm(0.5 * (term1 - term2))

    ElementaryOperator.define("op_create", [0], matrices._create)
    ElementaryOperator.define("op_annihilate", [0], matrices._annihilate)
    ElementaryOperator.define("op_number", [0], lambda dim: numpy.diag(numpy.arange(dim, dtype=numpy.complex128)))
    ElementaryOperator.define("op_parity", [0], lambda dim: numpy.diag([(-1.+0j)**i for i in range(dim)]))
    ElementaryOperator.define("op_displace", [0], matrices._displace)
    ElementaryOperator.define("op_squeeze", [0], matrices._squeeze)
    ElementaryOperator.define("op_position", [0], matrices._position)
    ElementaryOperator.define("op_momentum", [0], matrices._momentum)

    @classmethod
    def const(cls, constant_value: NumericType) -> ScalarOperator:
        return ScalarOperator.const(constant_value)
    @classmethod
    def zero(cls, degrees: list[int] | int = []) -> ScalarOperator | ElementaryOperator | ProductOperator:
        if isinstance(degrees, int): return ElementaryOperator.zero(degrees)
        elif len(degrees) == 0: return ScalarOperator.const(0)
        elif len(degrees) == 1: return ElementaryOperator.zero(degrees[0])
        else: return ProductOperator([ElementaryOperator.zero(degree) for degree in degrees])
    @classmethod
    def identity(cls, degrees: list[int] | int = []) -> ScalarOperator | ElementaryOperator | ProductOperator:
        if isinstance(degrees, int): return ElementaryOperator.identity(degrees)
        elif len(degrees) == 0: return ScalarOperator.const(1)
        elif len(degrees) == 1: return ElementaryOperator.identity(degrees[0])
        else: return ProductOperator([ElementaryOperator.identity(degree) for degree in degrees])
    @classmethod
    def create(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_create", [degree])
    @classmethod
    def annihilate(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_annihilate", [degree])
    @classmethod
    def number(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_number", [degree])
    @classmethod
    def parity(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_parity", [degree])
    @classmethod
    def displace(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_displace", [degree])
    @classmethod
    def squeeze(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_squeeze", [degree])
    @classmethod
    def position(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_position", [degree])
    @classmethod
    def momentum(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_momentum", [degree])

class pauli:
    ElementaryOperator.define("pauli_x", [2], lambda: numpy.array([[0,1],[1,0]], dtype=numpy.complex128))
    ElementaryOperator.define("pauli_y", [2], lambda: numpy.array([[0,1j],[-1j,0]], dtype=numpy.complex128))
    ElementaryOperator.define("pauli_z", [2], lambda: numpy.array([[1,0],[0,-1]], dtype=numpy.complex128))
    ElementaryOperator.define("pauli_i", [2], lambda: numpy.array([[1,0],[0,1]], dtype=numpy.complex128))
    ElementaryOperator.define("pauli_plus", [2], lambda: numpy.array([[0,0],[1,0]], dtype=numpy.complex128))
    ElementaryOperator.define("pauli_minus", [2], lambda: numpy.array([[0,1],[0,0]], dtype=numpy.complex128))

    @classmethod
    def x(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_x", [degree])
    @classmethod
    def y(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_y", [degree])
    @classmethod
    def z(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_z", [degree])
    @classmethod
    def i(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_i", [degree])
    @classmethod
    def plus(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_plus", [degree])
    @classmethod
    def minus(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("pauli_minus", [degree])


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


# To be implemented in C++ and bindings will be generated.
# If multiple initial states were passed, a sequence of evolution results is returned.
class EvolveResult:
    """
    Stores the execution data from an invocation of `cudaq.evolve`.
    """

    # Shape support in the type annotation for ndarray data type is still under development:
    # https://github.com/numpy/numpy/issues/16544
    def __init__(self: EvolveResult, 
                 state: cudaq.State | Iterable[cudaq.State],
                 expectation: Optional[NDArray[numpy.complexfloating] | Iterable[NDArray[numpy.complexfloating]]] = None) -> None:
        """
        Instantiates an EvolveResult representing the output generated when evolving a single 
        initial state under a set of operators. See `cudaq.evolve` for more detail.

        Arguments:
            state: A single state or a sequence of states of a quantum system. If a single 
                state is given, it represents the final state of the system after time evolution.
                If a sequence of states are given, they represent the state of the system after
                each steps in the schedule specified in the call to `cudaq.evolve`.
            expectation: A single one-dimensional array of complex values or a sequence of 
                one-dimensional arrays of complex values representing the expectation values
                computed during the evolution. If a single array is provided, it contains the
                expectation values computed at the end of the evolution. If a sequence of arrays
                is given, they represent the expectation values computed at each step in the 
                schedule passed to `cudaq.evolve`.
        """
        # This implementation is just a mock up - probably not very robust.
        *_, final_state = iter(state) # assumes cudaq.State is iterable - check if the type check here works
        if isinstance(final_state, cudaq.State):
            self._states = state
            self._final_state = final_state
        else:
            self._states = None
            self._final_state = state
        if expectation is None:
            self._expectation_values = None
            self._final_expectation : Optional[NDArray[numpy.complexfloating]] = None
        else:
            *_, final_expectation = iter(expectation)
            if isinstance(final_expectation, numpy.complexfloating):
                self._expectation_values = None
                self._final_expectation = expectation # type: ignore
            else:
                if self._states is None:
                    raise ValueError("intermediate states were defined but no intermediate expectation values are provided")
                self._expectation_values = expectation
                self._final_expectation = final_expectation

    @property
    def intermediate_states(self: EvolveResult) -> Optional[Iterable[cudaq.State]]:
        """
        Stores all intermediate states, meaning the state after each step in a defined 
        schedule, produced by a call to `cudaq.evolve`, including the final state. 
        This property is only populated if saving intermediate results was requested in the 
        call to `cudaq.evolve`.
        """
        return self._states

    @property
    def final_state(self: EvolveResult) -> cudaq.State:
        """
        Stores the final state produced by a call to `cudaq.evolve`.
        Represent the state of a quantum system after time evolution under a set of 
        operators, see the `cudaq.evolve` documentation for more detail.
        """
        return self._final_state

    @property
    def expectation_values(self: EvolveResult) -> Optional[Iterable[NDArray[numpy.complexfloating]]]:
        """
        Stores the expectation values at each step in the schedule produced by a call to 
        `cudaq.evolve`, including the final expectation values. Each expectation value 
        corresponds to one observable provided in the `cudaq.evolve` call. 
        This property is only populated if saving intermediate results was requested in the 
        call to `cudaq.evolve`. This value will be None if no intermediate results were 
        requested, or if no observables were specified in the call to `cudaq.evolve`.
        """
        return self._expectation_values

    @property
    def final_expectation(self: EvolveResult) -> Optional[NDArray[numpy.complexfloating]]:
        """
        Stores the final expectation values produced by a call to `cudaq.evolve`.
        Each expectation value corresponds to one observable provided in the `cudaq.evolve` call. 
        This value will be None if no observables were specified in the call to `cudaq.evolve`.
        """
        return self._final_expectation

# To be implemented in C++ and bindings will be generated.
class AsyncEvolveResult:
    """
    Stores the execution data from an invocation of `cudaq.evolve_async`.
    """

    def __init__(self: AsyncEvolveResult, handle: str) -> None:
        """
        Creates a class instance that can be used to retrieve the evolution
        result produces by an calling the asynchronously executing function
        `cudaq.evolve_async`. It models a future-like type whose 
        `EvolveResult` may be accessed via an invocation of the `get`
        method. 
        """
        raise NotImplementedError()

    def get(self: AsyncEvolveResult) -> EvolveResult:
        """
        Retrieves the `EvolveResult` from the asynchronous evolve execution.
        This causes the current thread to wait until the time evolution
        execution has completed. 
        """
        raise NotImplementedError()

    def __str__(self: AsyncEvolveResult) -> str:
        raise NotImplementedError()


# Top level API for the CUDA-Q master equation solver.
def evolve(hamiltonian: Operator, 
           dimensions: Mapping[int, int], 
           schedule: Schedule,
           initial_state: cudaq.State | Iterable[cudaq.States],
           collapse_operators: Iterable[Operator] = [],
           observables: Iterable[Operator] = [], 
           store_intermediate_results = False) -> EvolveResult | Iterable[EvolveResult]:
    """
    Computes the time evolution of one or more initial state(s) under the defined 
    operators. 

    Arguments:
        hamiltonian: Operator that describes the behavior of a quantum system
            without noise.
        dimensions: A mapping that specifies the number of levels, that is
            the dimension, of each degree of freedom that any of the operator 
            arguments acts on.
        schedule: An iterable that generates a mapping of keyword arguments 
            to their respective value. The keyword arguments are the parameters
            needed to evaluate any of the operators passed to `evolve`.
            All required parameters for evaluating an operator and their
            documentation, if available, can be queried by accessing the
            `parameter` property of the operator.
        initial_state: A single state or a sequence of states of a quantum system.
        collapse_operators: A sequence of operators that describe the influence of 
            noise on the quantum system.
        observables: A sequence of operators for which to compute their expectation
            value during evolution. If `store_intermediate_results` is set to True,
            the expectation values are computed after each step in the schedule, 
            and otherwise only the final expectation values at the end of the 
            evolution are computed.
    
    Returns:
        A single evolution result if a single initial state is provided, or a sequence
        of evolution results representing the data computed during the evolution of each
        initial state. See `EvolveResult` for more information about the data computed
        during evolution.
    """
    raise NotImplementedError()

def evolve_async(hamiltonian: Operator, 
                 dimensions: Mapping[int, int], 
                 schedule: Schedule,
                 initial_state: cudaq.State | Iterable[cudaq.States],
                 collapse_operators: Iterable[Operator] = [],
                 observables: Iterable[Operator] = [], 
                 store_intermediate_results = False) -> AsyncEvolveResult | Iterable[AsyncEvolveResult]:
    """
    Asynchronously computes the time evolution of one or more initial state(s) 
    under the defined operators. See `cudaq.evolve` for more details about the
    parameters passed here.
    
    Returns:
        The handle to a single evolution result if a single initial state is provided, 
        or a sequence of handles to the evolution results representing the data computed 
        during the evolution of each initial state. See the `EvolveResult` for more 
        information about the data computed during evolution.
    """
    raise NotImplementedError()

'''
dims = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}

print(f'pauliX(1): {pauli.x(1).to_matrix(dims)}')
print(f'pauliY(2): {pauli.y(2).to_matrix(dims)}')

print(f'pauliZ(0) * pauliZ(0): {(pauli.z(0) * pauli.z(0)).to_matrix(dims)}')
print(f'pauliZ(0) * pauliZ(1): {(pauli.z(0) * pauli.z(1)).to_matrix(dims)}')
print(f'pauliZ(0) * pauliY(1): {(pauli.z(0) * pauli.y(1)).to_matrix(dims)}')

op1 = ProductOperator([pauli.x(0), pauli.i(1)])
op2 = ProductOperator([pauli.i(0), pauli.x(1)])
print(f'pauliX(0) + pauliX(1): {op1.to_matrix(dims) + op2.to_matrix(dims)}')
op3 = ProductOperator([pauli.x(1), pauli.i(0)])
op4 = ProductOperator([pauli.i(1), pauli.x(0),])
print(f'pauliX(1) + pauliX(0): {op1.to_matrix(dims) + op2.to_matrix(dims)}')

print(f'pauliX(0) + pauliX(1): {(pauli.x(0) + pauli.x(1)).to_matrix(dims)}')
print(f'pauliX(0) * pauliX(1): {(pauli.x(0) * pauli.x(1)).to_matrix(dims)}')
print(f'pauliX(0) * pauliI(1) * pauliI(0) * pauliX(1): {(op1 * op2).to_matrix(dims)}')

print(f'pauliX(0) * pauliI(1): {op1.to_matrix(dims)}')
print(f'pauliI(0) * pauliX(1): {op2.to_matrix(dims)}')
print(f'pauliX(0) * pauliI(1) + pauliI(0) * pauliX(1): {(op1 + op2).to_matrix(dims)}')

op5 = pauli.x(0) * pauli.x(1)
op6 = pauli.z(0) * pauli.z(1)
print(f'pauliX(0) * pauliX(1): {op5.to_matrix(dims)}')
print(f'pauliZ(0) * pauliZ(1): {op6.to_matrix(dims)}')
print(f'pauliX(0) * pauliX(1) + pauliZ(0) * pauliZ(1): {(op5 + op6).to_matrix(dims)}')

op7 = pauli.x(0) + pauli.x(1)
op8 = pauli.z(0) + pauli.z(1)
print(f'pauliX(0) + pauliX(1): {op7.to_matrix(dims)}')
print(f'pauliZ(0) + pauliZ(1): {op8.to_matrix(dims)}')
print(f'pauliX(0) + pauliX(1) + pauliZ(0) + pauliZ(1): {(op7 + op8).to_matrix(dims)}')
print(f'(pauliX(0) + pauliX(1)) * (pauliZ(0) + pauliZ(1)): {(op7 * op8).to_matrix(dims)}')

print(f'pauliX(0) * (pauliZ(0) + pauliZ(1)): {(pauli.x(0) * op8).to_matrix(dims)}')
print(f'(pauliZ(0) + pauliZ(1)) * pauliX(0): {(op8 * pauli.x(0)).to_matrix(dims)}')

op9 = pauli.z(1) + pauli.z(2)
print(f'(pauliX(0) + pauliX(1)) * pauliI(2): {numpy.kron(op7.to_matrix(dims), pauli.i(2).to_matrix(dims))}')
print(f'(pauliX(0) + pauliX(1)) * pauliI(2): {(op7 * pauli.i(2)).to_matrix(dims)}')
print(f'(pauliX(0) + pauliX(1)) * pauliI(2): {(pauli.i(2) * op7).to_matrix(dims)}')
print(f'pauliI(0) * (pauliZ(1) + pauliZ(2)): {numpy.kron(pauli.i(0).to_matrix(dims), op9.to_matrix(dims))}')
print(f'(pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)): {(op7 * op9).to_matrix(dims)}')

so0 = ScalarOperator(lambda: 1.0j)
print(f'Scalar op (t -> 1.0)(): {so0.to_matrix()}')

so1 = ScalarOperator(lambda t: t)
print(f'Scalar op (t -> t)(1.): {so1.to_matrix(t = 1.0)}')
print(f'Trivial prod op (t -> t)(1.): {(ProductOperator([so1])).to_matrix({}, t = 1.)}')
print(f'Trivial prod op (t -> t)(2.): {(ProductOperator([so1])).to_matrix({}, t = 2.)}')

print(f'(t -> t)(1j) * pauliX(0): {(so1 * pauli.x(0)).to_matrix(dims, t = 1j)}')
print(f'pauliX(0) * (t -> t)(1j): {(pauli.x(0) * so1).to_matrix(dims, t = 1j)}')
print(f'pauliX(0) + (t -> t)(1j): {(pauli.x(0) + so1).to_matrix(dims, t = 1j)}')
print(f'(t -> t)(1j) + pauliX(0): {(so1 + pauli.x(0)).to_matrix(dims, t = 1j)}')
print(f'pauliX(0) + (t -> t)(1j): {(pauli.x(0) + so1).to_matrix(dims, t = 1j)}')
print(f'(t -> t)(1j) + pauliX(0): {(so1 + pauli.x(0)).to_matrix(dims, t = 1j)}')
print(f'(t -> t)(2.) * (pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)): {(so1 * op7 * op9).to_matrix(dims, t = 2.)}')
print(f'(pauliX(0) + pauliX(1)) * (t -> t)(2.) * (pauliZ(1) + pauliZ(2)): {(op7 * so1 * op9).to_matrix(dims, t = 2.)}')
print(f'(pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)) * (t -> t)(2.): {(op7 * op9 * so1).to_matrix(dims, t = 2.)}')

op10 = so1 * pauli.x(0)
so1.generator = lambda t: 1./t
print(f'(t -> 1/t)(2) * pauliX(0): {op10.to_matrix(dims, t = 2.)}')
so1_gen2 = so1.generator
so1.generator = lambda t: so1_gen2(2*t)
print(f'(t -> 1/(2t))(2) * pauliX(0): {op10.to_matrix(dims, t = 2.)}')
so1.generator = lambda t: so1_gen2(t)
print(f'(t -> 1/t)(2) * pauliX(0): {op10.to_matrix(dims, t = 2.)}')

so2 = ScalarOperator(lambda t: t**2)
op11 = pauli.z(1) * so2
print(f'pauliZ(0) * (t -> t^2)(2.): {op11.to_matrix(dims, t = 2.)}')

so3 = ScalarOperator(lambda t: 1./t)
so4 = ScalarOperator(lambda t: t**2)
print(f'((t -> 1/t) * (t -> t^2))(2.): {(so3 * so4).to_matrix(t = 2.)}')
so5 = so3 + so4
so3.generator = lambda field: 1./field
print(f'((f -> 1/f) + (t -> t^2))(f=2, t=1.): {so5.to_matrix(t = 1., field = 2)}')

def generator(field, **kwargs):
    print(f'generator got kwargs: {kwargs}')
    return field

so3.generator = generator
print(f'((f -> f) + (t -> t^2))(f=3, t=2): {so5.to_matrix(field = 3, t = 2, dummy = 10)}')

so6 = ScalarOperator(lambda foo, *, bar: foo * bar)
print(f'((f,t) -> f*t)(f=3, t=2): {so6.to_matrix(foo = 3, bar = 2, dummy = 10)}')
so7 = ScalarOperator(lambda foo, *, bar, **kwargs: foo * bar)
print(f'((f,t) -> f*t)(f=3, t=2): {so6.to_matrix(foo = 3, bar = 2, dummy = 10)}')

def get_parameter_value(parameter_name: str, time: float):
    match parameter_name:
        case "foo": return time
        case "bar": return 2 * time
        case _: raise NotImplementedError(f'No value defined for parameter {parameter_name}.')

schedule = Schedule([0.0, 0.5, 1.0], so6.parameters, get_parameter_value)
for parameters in schedule:
    print(f'step {schedule.current_step}')
    print(f'((f,t) -> f*t)({parameters}): {so6.to_matrix({}, **parameters)}')

print(f'(pauliX(0) + i*pauliY(0))/2: {0.5 * (pauli.x(0) + operators.const(1j) * pauli.y(0)).to_matrix(dims)}')
print(f'pauli+(0): {pauli.plus(0).to_matrix(dims)}')
print(f'(pauliX(0) - i*pauliY(0))/2: {0.5 * (pauli.x(0) - operators.const(1j) * pauli.y(0)).to_matrix(dims)}')
print(f'pauli-(0): {pauli.minus(0).to_matrix(dims)}')

op12 = operators.squeeze(0) + operators.displace(0)
print(f'create<3>(0): {operators.create(0).to_matrix({0:3})}')
print(f'annihilate<3>(0): {operators.annihilate(0).to_matrix({0:3})}')
print(f'squeeze<3>(0)[squeezing = 0.5]: {operators.squeeze(0).to_matrix({0:3}, squeezing=0.5)}')
print(f'displace<3>(0)[displacement = 0.5]: {operators.displace(0).to_matrix({0:3}, displacement=0.5)}')
print(f'(squeeze<3>(0) + displace<3>(0))[squeezing = 0.5, displacement = 0.5]: {op12.to_matrix({0:3}, displacement=0.5, squeezing=0.5)}')
print(f'squeeze<4>(0)[squeezing = 0.5]: {operators.squeeze(0).to_matrix({0:4}, squeezing=0.5)}')
print(f'displace<4>(0)[displacement = 0.5]: {operators.displace(0).to_matrix({0:4}, displacement=0.5)}')
print(f'(squeeze<4>(0) + displace<4>(0))[squeezing = 0.5, displacement = 0.5]: {op12.to_matrix({0:4}, displacement=0.5, squeezing=0.5)}')

so8 = ScalarOperator(lambda my_param: my_param - 1)
so9 = so7 * so8
print(f'parameter descriptions: {operators.squeeze(0).parameters}')
print(f'parameter descriptions: {op12.parameters}')
print(f'parameter descriptions: {(so7 + so8).parameters}')
print(f'parameter descriptions: {(operators.squeeze(0) * operators.displace(0)).parameters}')
print(f'parameter descriptions: {so9.parameters}')
so7.generator = lambda new_parameter: 1.0
print(f'parameter descriptions: {so9.parameters}')
so9.generator = lambda reset: reset
print(f'parameter descriptions: {so9.parameters}')

def all_zero(sure, args):
    """Some args documentation.
    Args:

      sure (:obj:`int`, optional): my docs for sure
      args: Description of `args`. Multiple
            lines are supported.
    Returns:
      Something that for sure is correct.
    """
    if sure: return 0
    else: return 1

print(f'parameter descriptions: {(ScalarOperator(all_zero)).parameters}')

scop = operators.const(2)
elop = operators.identity(1)
print(f"arithmetics: {scop.to_matrix(dims)}")
print(f"arithmetics: {elop.to_matrix(dims)}")
print(f"arithmetics: {(scop * elop).to_matrix(dims)}")
print(f"arithmetics: {(elop * scop).to_matrix(dims)}")
print(f"arithmetics: {(scop + elop).to_matrix(dims)}")
print(f"arithmetics: {(elop + scop).to_matrix(dims)}")
print(f"arithmetics: {(scop - elop).to_matrix(dims)}")
print(f"arithmetics: {(elop - scop).to_matrix(dims)}")
print(f"arithmetics: {((scop * elop) * scop).to_matrix(dims)}")
print(f"arithmetics: {(scop * (scop * elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop * elop) * elop).to_matrix(dims)}")
print(f"arithmetics: {(elop * (scop * elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop * elop) + scop).to_matrix(dims)}")
print(f"arithmetics: {(scop + (scop * elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop * elop) + elop).to_matrix(dims)}")
print(f"arithmetics: {(elop + (scop * elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop * elop) - scop).to_matrix(dims)}")
print(f"arithmetics: {(scop - (scop * elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop * elop) - elop).to_matrix(dims)}")
print(f"arithmetics: {(elop - (scop * elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop + elop) * scop).to_matrix(dims)}")
print(f"arithmetics: {(scop * (scop + elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop + elop) * elop).to_matrix(dims)}")
print(f"arithmetics: {(elop * (scop + elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop - elop) * scop).to_matrix(dims)}")
print(f"arithmetics: {(scop * (scop - elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop - elop) * elop).to_matrix(dims)}")
print(f"arithmetics: {(elop * (scop - elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop + elop) + scop).to_matrix(dims)}")
print(f"arithmetics: {(scop + (scop + elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop + elop) + elop).to_matrix(dims)}")
print(f"arithmetics: {(elop + (scop + elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop - elop) - scop).to_matrix(dims)}")
print(f"arithmetics: {(scop - (scop - elop)).to_matrix(dims)}")
print(f"arithmetics: {((scop - elop) - elop).to_matrix(dims)}")
print(f"arithmetics: {(elop - (scop - elop)).to_matrix(dims)}")

opprod = operators.create(0) * operators.annihilate(0)
opsum = operators.create(0) + operators.annihilate(0)
for arith in [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]:
    print(f"testing {arith} for ScalarOperator")
    print(f"arithmetics: {arith(scop, 2).to_matrix(dims)}")
    print(f"arithmetics: {arith(scop, 2.5).to_matrix(dims)}")
    print(f"arithmetics: {arith(scop, 2j).to_matrix(dims)}")
    print(f"arithmetics: {arith(2, scop).to_matrix(dims)}")
    print(f"arithmetics: {arith(2.5, scop).to_matrix(dims)}")
    print(f"arithmetics: {arith(2j, scop).to_matrix(dims)}")

for op in [elop, opprod, opsum]:
    for arith in [operator.add, operator.sub, operator.mul]:
        print(f"testing {arith} for {type(op)}")
        print(f"arithmetics: {arith(op, 2).to_matrix(dims)}")
        print(f"arithmetics: {arith(op, 2.5).to_matrix(dims)}")
        print(f"arithmetics: {arith(op, 2j).to_matrix(dims)}")
        print(f"arithmetics: {arith(2, op).to_matrix(dims)}")
        print(f"arithmetics: {arith(2.5, op).to_matrix(dims)}")
        print(f"arithmetics: {arith(2j, op).to_matrix(dims)}")

print(operators.const(2))
print(ScalarOperator(lambda alpha: 2*alpha))
print(ScalarOperator(all_zero))
print(pauli.x(0))
print(2 * pauli.x(0))
print(pauli.x(0) + 2)
print(operators.squeeze(0))
print(operators.squeeze(0) * operators.displace(1))
print(operators.squeeze(0) + operators.displace(1) * 5)
print(pauli.x(0) - 2)
print(pauli.x(0) - pauli.y(1))
print(pauli.x(0).degrees)
print((pauli.x(2) * pauli.y(0)).degrees)
print((pauli.x(2) + pauli.y(0)).degrees)

print(ScalarOperator.const(5) == ScalarOperator.const(5))
print(ScalarOperator.const(5) == ScalarOperator.const(5+0j))
print(ScalarOperator.const(5) == ScalarOperator.const(5j))
print(ScalarOperator(lambda: 5) == ScalarOperator.const(5))
print(ScalarOperator(lambda: 5) == ScalarOperator(lambda: 5))
gen = lambda: 5
so10 = ScalarOperator(gen)
so11 = ScalarOperator(lambda: 5)
print(so10 == so11)
print(so10 == ScalarOperator(gen))
so11.generator = gen
print(so10 == so11)
print(ElementaryOperator.identity(1) * so10 == ElementaryOperator.identity(1) * so11)
print(ElementaryOperator.identity(1) + so10 == ElementaryOperator.identity(1) + so11)
print(pauli.x(1) + pauli.y(1) == pauli.y(1) + pauli.x(1))
print(pauli.x(1) * pauli.y(1) == pauli.y(1) * pauli.x(1))
print(pauli.x(0) + pauli.y(1) == pauli.y(1) + pauli.x(0))
print(pauli.x(0) * pauli.y(1) == pauli.y(1) * pauli.x(0))
print(opprod == opprod)
print(opprod * so10 == so10 * opprod)
print(opprod + so10 == so10 + opprod)
print(ScalarOperator.const(10) * opprod == opprod * ScalarOperator.const(10.0))
print(ScalarOperator.const(10) + opprod == opprod + ScalarOperator.const(10.0))
paulizy = lambda i,j: pauli.z(i) * pauli.y(j)
paulixy = lambda i,j: pauli.x(i) * pauli.y(j)
print(paulixy(0,0) + paulizy(0,0) == paulizy(0,0) + paulixy(0,0))
print(paulixy(0,0) * paulizy(0,0) == paulizy(0,0) * paulixy(0,0))
print(paulixy(1,1) * paulizy(0,0) == paulizy(0,0) * paulixy(1,1)) # We have multiple terms acting on the same degree of freedom, so we don't try to reorder here.
print(paulixy(1,2) * paulizy(3,4) == paulizy(3,4) * paulixy(1,2))
'''
