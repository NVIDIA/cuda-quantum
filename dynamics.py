import inspect, itertools, numpy, os, re, scipy
from numbers import Number
from typing import Any, Callable, Iterable, Iterator
from numpy.typing import NDArray

class _OperatorHelpers:

    # Helper function used by all operator classes to return a dictionary with the 
    # used parameters and their respective description as defined in a doc comment.
    def aggregate_parameters(dicts: Iterable[dict[str, str]]):
        param_descriptions = {}
        for dict in dicts:
            for key in dict:
                existing_desc = param_descriptions.get(key)
                new_desc = dict[key]
                has_existing_desc = existing_desc is not None and existing_desc != ""
                has_new_desc = new_desc != ""
                if has_existing_desc and has_new_desc:
                    param_descriptions[key] = existing_desc + f'{os.linesep}---{os.linesep}' + new_desc
                elif has_existing_desc:
                    param_descriptions[key] = existing_desc
                else: 
                    param_descriptions[key] = new_desc
        return param_descriptions
    
    def parameter_docs(param_name: str, docs: str):
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
            split = re.split(keyword_pattern("Args"), docs, flags=re.MULTILINE)
            if len(split) > 1:
                match = re.search(param_pattern(param_name), split[1], re.MULTILINE)
                if match != None:
                    param_docs = match.group(2) + split[1][match.end(2):]
                    match = re.search(param_pattern("\S*?"), param_docs, re.MULTILINE)
                    if match != None:
                        param_docs = param_docs[:match.start(0)]
                    param_docs = re.sub(r'\s+', ' ', param_docs)
        except: pass
        return param_docs.strip()

    # Extracts the positional argument and keyword only arguments 
    # for the given function from the passed kwargs. 
    def args_from_kwargs(fct, **kwargs):
        arg_spec = inspect.getfullargspec(fct)
        signature = inspect.signature(fct)
        if arg_spec.varargs is not None:
            raise ValueError("cannot extract arguments for a function with a *args argument")
        consumes_kwargs = arg_spec.varkw is not None
        if consumes_kwargs:
            kwargs = kwargs.copy() # We will modify and return a copy

        def find_in_kwargs(arg_name: str):
            # Try to get the argument from the kwargs passed to concretize.
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

    # Generates all possible states for the given dimensions ordered according to 
    # the list of degrees (ordering is relevant if dimensions differ).
    def generate_all_states(degrees: list[int], dimensions: dict[int, int]):
        if len(degrees) == 0:
            return []
        states = [[str(state)] for state in range(dimensions[degrees[0]])]
        for d in degrees[1:]:
            prod = itertools.product(states, [str(state) for state in range(dimensions[d])])
            states = [current + [new] for current, new in prod]
        return [''.join(state) for state in states]


class ScalarOperator:
    pass

class ElementaryOperator:
    pass

class ProductOperator:
    pass

class OperatorSum:
    pass

class OperatorSum:

    def __init__(self: OperatorSum, terms: list[ProductOperator]):
        if len(terms) == 0:
            raise ValueError("need at least one term")
        self._terms = terms

    @property
    def parameters(self: OperatorSum) -> dict[str, str]:
        return _OperatorHelpers.aggregate_parameters([term.parameters for term in self._terms])

    def concretize(self: OperatorSum, dimensions: dict[int, int], **kwargs) -> NDArray[complex]:
        degrees = set([degree for term in self._terms for op in term._operators for degree in op._degrees])
        padded_terms = [] # We need to make sure all matrices are of the same size to sum them up.
        for term in self._terms:
            for degree in degrees:
                if not degree in [op_degree for op in term._operators for op_degree in op._degrees]:
                    term *= operators.identity(degree)
            padded_terms.append(term)
        return sum([term.concretize(dimensions, **kwargs) for term in padded_terms])

    def __mul__(self: OperatorSum, other) -> OperatorSum:
        if not (isinstance(other, OperatorSum) or isinstance(other, Number)):
            return NotImplemented
        elif type(other) == OperatorSum:
            return OperatorSum([self_term * other_term for self_term in self._terms for other_term in other._terms])
        return OperatorSum([self_term * other for self_term in self._terms])

    def __rmul__(self: OperatorSum, other) -> OperatorSum:
        # We only need to handle multiplication with Numbers here, 
        # since everything else is covered without right multiplication.
        if isinstance(other, Number):
            return OperatorSum([self_term * other for self_term in self._terms])
        return NotImplemented

    def __add__(self: OperatorSum, other) -> OperatorSum:
        if not isinstance(other, OperatorSum):
            return NotImplemented        
        elif type(other) == OperatorSum:
            return OperatorSum(self._terms + other._terms)
        return other + self # Operator addition is commutative.

    def __sub__(self: OperatorSum, other) -> OperatorSum:
        return self + operators.const(-1) * other

class ProductOperator(OperatorSum):

    def __init__(self: ProductOperator, operators : list[ElementaryOperator | ScalarOperator]):
        if len(operators) == 0:
            raise ValueError("need at least one operator")
        self._operators = operators
        super().__init__([self])

    @property
    def parameters(self: ProductOperator) -> dict[str, str]:
        return _OperatorHelpers.aggregate_parameters([operator.parameters for operator in self._operators])

    def concretize(self: ProductOperator, dimensions: dict[int, int], **kwargs) -> NDArray[complex]:
        def padded_matrix(op: ElementaryOperator | ScalarOperator, degrees: list[int]):
            op_matrix = op.concretize(dimensions, **kwargs)
            op_degrees = op._degrees.copy() # Determines the initial qubit ordering of op_matrix.
            for degree in degrees:
                if not degree in [d for d in op._degrees]:
                    op_matrix = numpy.kron(op_matrix, operators.identity(degree).concretize(dimensions, **kwargs))
                    op_degrees.append(degree)
            # Need to permute the matrix such that the qubit ordering of all matrices is the same.
            if op_degrees != degrees:
                # There may be a more efficient way, but I needed something correct first.
                states = _OperatorHelpers.generate_all_states(degrees, dimensions)
                indices = dict([(d, idx) for idx, d in enumerate(degrees)])
                reordering = [indices[op_degree] for op_degree in op_degrees]
                # [degrees[i] for i in reordering] produces op_degrees
                op_states = [''.join([state[i] for i in reordering]) for state in states]
                state_indices = dict([(state, idx) for idx, state in enumerate(states)])
                permutation = [state_indices[op_state] for op_state in op_states]
                # [states[i] for i in permutation] produces op_states
                for i in range(numpy.size(op_matrix, 1)):
                    op_matrix[:,i] = op_matrix[permutation,i]
                for i in range(numpy.size(op_matrix, 0)):
                    op_matrix[i,:] = op_matrix[i,permutation]
            return op_matrix
        
        degrees = list(set([degree for op in self._operators for degree in op._degrees]))
        degrees.sort() # This sorting determines the qubit ordering of the final matrix.
        # FIXME: check endianness ... (in the sorting/states here, and in the matrix definitions)
        matrix = padded_matrix(self._operators[0], degrees)
        for op in self._operators[1:]:
            matrix = numpy.dot(matrix, padded_matrix(op, degrees))
        return matrix

    def __mul__(self: ProductOperator, other) -> ProductOperator | OperatorSum:
        if not (isinstance(other, OperatorSum) or isinstance(other, Number)):
            return NotImplemented
        elif type(other) == ProductOperator:
            return ProductOperator(self._operators + other._operators)
        elif type(other) == OperatorSum: # Only create OperatorSum if needed.
            return OperatorSum([self]) * other
        elif isinstance(other, Number):
            return ProductOperator(self._operators + [operators.const(other)])
        return self * ProductOperator([other])
    
    def __rmul__(self: ProductOperator, other) -> ProductOperator:
        # We only need to handle multiplication with Numbers here, 
        # since everything else is covered without right multiplication.
        if isinstance(other, Number):
            return ProductOperator([operators.const(other)] + self._operators)
        return NotImplemented

    def __add__(self: ProductOperator, other) -> OperatorSum:
        if not isinstance(other, OperatorSum):
            return NotImplemented
        elif type(other) == ProductOperator:
            return OperatorSum([self, other])
        return OperatorSum([self]) + other

    def __sub__(self: ProductOperator, other) -> OperatorSum:
        return self + operators.const(-1) * other

class ElementaryOperator(ProductOperator):
    _ops = {} # Contains the generator for each defined ElementaryOperator.
    _parameter_info = {} # Information about all required parameters for each ElementaryOperator.

    # The Callable `create` that is passed here may take any number and types of 
    # arguments and must return a NDArray[complex]. Each argument must be passed
    # as keyword arguments when concretizing any operator that involves this built-in.
    # Note that the dimensions passed to the create function are automatically validated 
    # against the expected/supported dimensions passed to `add_operator`. There is hence 
    # no need to validate the dimensions as part of the `create` function. 
    # A negative or zero value for one (or more) of the expected dimensions indicates 
    # that the matrix/operator is defined for any dimension of the corresponding degree 
    # of freedom. If the operator definition depends on the dimensions, then the 
    # `create` function must take an argument called `dimensions` (or `dims` for short),
    # or `dimension` (or `dim` for short) if it is just one. The given list of dimensions, 
    # or its only entry if it is a list of length one, will then be automatically 
    # forwarded as argument to `create`.
    @classmethod
    def define(cls, op_id: str, expected_dimensions: list[int], create: Callable):
        forwarded_as_kwarg = [["dimensions", "dims"], ["dimension", "dim"]]
        def with_dimension_check(creation: Callable, dimensions: list[int], **kwargs) -> Callable:
            # Passing a value 0 for one of the expected dimensions indicates that
            # the creation function can be invoked with any value for that dimension.
            if any([expected > 0 and dimensions[i] != expected for i, expected in enumerate(expected_dimensions)]):
                raise ValueError(f'no built-in operator {op_id} has been defined '\
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
                raise TypeError("operator concretization must return a 'NDArray[complex]'")
            return evaluated
        generator = lambda dimensions, **kwargs: with_dimension_check(create, dimensions, **kwargs)
        parameters, forwarded = {}, list([keyword for group in forwarded_as_kwarg for keyword in group])
        arg_spec = inspect.getfullargspec(create)
        for pname in arg_spec.args + arg_spec.kwonlyargs:
            if not pname in forwarded:
                parameters[pname] = _OperatorHelpers.parameter_docs(pname, create.__doc__)
        cls._ops[op_id] = generator
        cls._parameter_info[op_id] = parameters

    def __init__(self: ElementaryOperator, operator_id: str, degrees: list[int]):
        if not operator_id in ElementaryOperator._ops:
            raise ValueError(f"no built-in operator '{operator_id}' has been defined")
        self._op_id = operator_id
        self._degrees = degrees
        self._degrees.sort() # sorting so that we have a unique ordering for builtin
        super().__init__([self])

    @property
    def parameters(self: ElementaryOperator) -> dict[str, str]:
        return ElementaryOperator._parameter_info[self._op_id]

    def concretize(self: ElementaryOperator, dimensions: dict[int, int], **kwargs) -> NDArray[complex]:
        missing_degrees = [degree not in dimensions for degree in self._degrees]
        if any(missing_degrees):
            raise ValueError(f'missing dimensions for degree(s) {[self._degrees[i] for i, x in enumerate(missing_degrees) if x]}')
        relevant_dimensions = [dimensions[degree] for degree in self._degrees]
        return ElementaryOperator._ops[self._op_id](relevant_dimensions, **kwargs)

    def __mul__(self: ElementaryOperator, other) -> ProductOperator:
        if not (isinstance(other, OperatorSum) or isinstance(other, Number)):
            return NotImplemented
        elif type(other) == ElementaryOperator:
            return ProductOperator([self, other])
        return ProductOperator([self]) * other

    def __rmul__(self: ElementaryOperator, other) -> ProductOperator:
        # We only need to handle multiplication with Numbers here, 
        # since everything else is covered without right multiplication.
        if isinstance(other, Number):
            return ProductOperator([self]) * other
        return NotImplemented

    def __add__(self: ElementaryOperator, other) -> OperatorSum:
        if not isinstance(other, OperatorSum):
            return NotImplemented
        elif type(other) == ElementaryOperator:
            op1 = ProductOperator([self])
            op2 = ProductOperator([other])
            return OperatorSum([op1, op2])
        return ProductOperator([self]) + other

    def __sub__(self: ElementaryOperator, other) -> OperatorSum:
        return self + operators.const(-1) * other

class ScalarOperator(ProductOperator):

    # The given generator may take any number and types of arguments, 
    # and must return a numeric value. Each argument must be passed
    # as keyword arguments when concretizing any operator that involves
    # this scalar operator.
    def __init__(self: ScalarOperator, generator: Callable):
        self._degrees = []
        self.generator = generator # The setter validates the generator and sets _parameter_info.
        super().__init__([self])

    @property
    def generator(self: ScalarOperator):
        return self._generator

    @generator.setter
    def generator(self: ScalarOperator, generator: Callable):
        # A variable number of arguments (i.e. *args) cannot be supported
        # for generators; it would prevent proper argument handling while 
        # supporting additions and multiplication of all kinds of operators.
        arg_spec = inspect.getfullargspec(generator)
        if arg_spec.varargs is not None:
            raise ValueError(f"generator for a '{type(self).__name__}' must not take *args")
        param_descriptions = {}
        for arg_name in arg_spec.args + arg_spec.kwonlyargs:
            param_descriptions[arg_name] = _OperatorHelpers.parameter_docs(arg_name, generator.__doc__)
        # We need to create a lambda to retrieve information about what
        # parameters are required to invoke the generator, to ensure that
        # the information accurately captures any updates to the generators
        # of sub-operators, just like the lambda in the add/multiply below.    
        self._parameter_info = lambda: param_descriptions
        self._generator = generator

    @property
    def parameters(self: ScalarOperator) -> dict[str, str]:
        return self._parameter_info()

    # The argument `dimensions` here is only passed for consistency with parent classes.
    def concretize(self: ScalarOperator, dimensions: dict[int, int] = None, **kwargs):
        generator_args, remaining_kwargs = _OperatorHelpers.args_from_kwargs(self._generator, **kwargs)
        evaluated = self._generator(*generator_args, **remaining_kwargs)
        if not isinstance(evaluated, Number):
            raise TypeError(f"generator of {type(self).__name__} must return a 'Number'")
        return evaluated

    def __mul__(self: ScalarOperator, other) -> ScalarOperator | ProductOperator:
        if not (isinstance(other, OperatorSum) or isinstance(other, Number)):
            return NotImplemented
        elif type(other) == ScalarOperator:
            generator = lambda **kwargs: self.concretize(**kwargs) * other.concretize(**kwargs)
            operator = ScalarOperator(generator)
            operator._parameter_info = lambda: _OperatorHelpers.aggregate_parameters([self._parameter_info(), other._parameter_info()])
            return operator
        elif isinstance(other, Number):
            generator = lambda **kwargs: other * self.concretize(**kwargs)
            operator = ScalarOperator(generator)
            operator._parameter_info = self._parameter_info
            return operator
        return ProductOperator([self]) * other

    def __rmul__(self: ScalarOperator, other) -> ProductOperator:
        # We only need to handle multiplication with Numbers here, 
        # since everything else is covered without right multiplication.
        if isinstance(other, Number):
            return self * other
        return NotImplemented

    def __add__(self: ScalarOperator, other) -> ScalarOperator | OperatorSum:
        if not isinstance(other, OperatorSum):
            return NotImplemented
        elif type(other) == ScalarOperator:
            generator = lambda **kwargs: self.concretize(**kwargs) + other.concretize(**kwargs)
            operator = ScalarOperator(generator)
            operator._parameter_info = lambda: _OperatorHelpers.aggregate_parameters([self._parameter_info(), other._parameter_info()])
            return operator
        return ProductOperator([self]) + other

    def __sub__(self: ScalarOperator, other) -> ScalarOperator | OperatorSum:
        return self + operators.const(-1) * other


# Operators as defined here (watch out of differences in convention): 
# https://www.dynamiqs.org/python_api/utils/operators/sigmay.html
class operators:

    def _create(dimension: int): 
        return numpy.diag(numpy.sqrt(numpy.arange(1, dimension, dtype=complex)), -1)
    def _annihilate(dimension: int): 
        return numpy.diag(numpy.sqrt(numpy.arange(1, dimension, dtype=complex)), 1)
    def _position(dimension: int):
        return 0.5 * (operators._create(dimension) + operators._annihilate(dimension))
    def _momentum(dimension: int):
        return 0.5j * (operators._create(dimension) - operators._annihilate(dimension))
    def _displace(dimension: int, displacement: complex):
        """Connects to the next available port.
        Args:
            displacement: Amplitude of the displacement operator.
                See also https://en.wikipedia.org/wiki/Displacement_operator.
        """
        term1 = displacement * operators._create(dimension)
        term2 = numpy.conjugate(displacement) * operators._annihilate(dimension)
        return scipy.linalg.expm(term1 - term2)
    def _squeeze(dimension: int, squeezing: complex):
        """Connects to the next available port.
        Args:
            squeezing: Amplitude of the squeezing operator.
                See also https://en.wikipedia.org/wiki/Squeeze_operator.
        """
        term1 = numpy.conjugate(squeezing) * numpy.linalg.matrix_power(operators._annihilate(dimension), 2)
        term2 = squeezing * numpy.linalg.matrix_power(operators._create(dimension), 2)
        return scipy.linalg.expm(0.5 * (term1 - term2))

    ElementaryOperator.define("op_zero", [0], lambda dim: numpy.zeros((dim, dim), dtype=complex))
    ElementaryOperator.define("op_identity", [0], lambda dim: numpy.diag(numpy.ones(dim, dtype=complex)))
    ElementaryOperator.define("op_create", [0], _create)
    ElementaryOperator.define("op_annihilate", [0], _annihilate)
    ElementaryOperator.define("op_number", [0], lambda dim: numpy.diag(numpy.arange(dim, dtype=complex)))
    ElementaryOperator.define("op_parity", [0], lambda dim: numpy.diag([(-1.+0j)**i for i in range(dim)]))
    ElementaryOperator.define("op_displace", [0], _displace)
    ElementaryOperator.define("op_squeeze", [0], _squeeze)
    ElementaryOperator.define("op_position", [0], _position)
    ElementaryOperator.define("op_momentum", [0], _momentum)

    @classmethod
    def const(cls, constant_value: complex) -> ScalarOperator:
        return ScalarOperator(lambda: constant_value)
    @classmethod
    def zero(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_zero", [degree])
    @classmethod
    def identity(cls, degree: int) -> ElementaryOperator:
        return ElementaryOperator("op_identity", [degree])
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
    ElementaryOperator.define("pauli_x", [2], lambda: numpy.array([[0,1],[1,0]], dtype=complex))
    ElementaryOperator.define("pauli_y", [2], lambda: numpy.array([[0,1j],[-1j,0]], dtype=complex))
    ElementaryOperator.define("pauli_z", [2], lambda: numpy.array([[1,0],[0,-1]], dtype=complex))
    ElementaryOperator.define("pauli_i", [2], lambda: numpy.array([[1,0],[0,1]], dtype=complex))
    ElementaryOperator.define("pauli_plus", [2], lambda: numpy.array([[0,0],[1,0]], dtype=complex))
    ElementaryOperator.define("pauli_minus", [2], lambda: numpy.array([[0,1],[0,0]], dtype=complex))

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
    
class Schedule:
    _iterator: Iterator[Any]
    _parameters: Iterable[str]
    _get_value: Callable[[str, Any], Any]
    _current_step: Any

    # The output type of the iterable steps must match the second argument of `get_value`.
    def __init__(self, steps: Iterable[Any], parameters: Iterable[str], get_value: Callable[[str, Any], Any]):
        self._iterator = iter(steps)
        self._parameters = parameters
        self._get_value = get_value
        self._current_step = None

    @property
    def current_step(self):
        return self._current_step
    
    def __iter__(self):
        return self
        
    def __next__(self):
        self._current_step = None # Set current_step to None when we reach the end of the iteration.
        self._current_step = next(self._iterator)
        kvargs = {}
        for parameter in self._parameters:
            kvargs[parameter] = self._get_value(parameter, self._current_step)
        return kvargs


dims = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}

print(f'pauliX(1): {pauli.x(1).concretize(dims)}')
print(f'pauliY(2): {pauli.y(2).concretize(dims)}')

print(f'pauliZ(0) * pauliZ(0): {(pauli.z(0) * pauli.z(0)).concretize(dims)}')
print(f'pauliZ(0) * pauliZ(1): {(pauli.z(0) * pauli.z(1)).concretize(dims)}')
print(f'pauliZ(0) * pauliY(1): {(pauli.z(0) * pauli.y(1)).concretize(dims)}')

op1 = ProductOperator([pauli.x(0), pauli.i(1)])
op2 = ProductOperator([pauli.i(0), pauli.x(1)])
print(f'pauliX(0) + pauliX(1): {op1.concretize(dims) + op2.concretize(dims)}')
op3 = ProductOperator([pauli.x(1), pauli.i(0)])
op4 = ProductOperator([pauli.i(1), pauli.x(0),])
print(f'pauliX(1) + pauliX(0): {op1.concretize(dims) + op2.concretize(dims)}')

print(f'pauliX(0) + pauliX(1): {(pauli.x(0) + pauli.x(1)).concretize(dims)}')
print(f'pauliX(0) * pauliX(1): {(pauli.x(0) * pauli.x(1)).concretize(dims)}')
print(f'pauliX(0) * pauliI(1) * pauliI(0) * pauliX(1): {(op1 * op2).concretize(dims)}')

print(f'pauliX(0) * pauliI(1): {op1.concretize(dims)}')
print(f'pauliI(0) * pauliX(1): {op2.concretize(dims)}')
print(f'pauliX(0) * pauliI(1) + pauliI(0) * pauliX(1): {(op1 + op2).concretize(dims)}')

op5 = pauli.x(0) * pauli.x(1)
op6 = pauli.z(0) * pauli.z(1)
print(f'pauliX(0) * pauliX(1): {op5.concretize(dims)}')
print(f'pauliZ(0) * pauliZ(1): {op6.concretize(dims)}')
print(f'pauliX(0) * pauliX(1) + pauliZ(0) * pauliZ(1): {(op5 + op6).concretize(dims)}')

op7 = pauli.x(0) + pauli.x(1)
op8 = pauli.z(0) + pauli.z(1)
print(f'pauliX(0) + pauliX(1): {op7.concretize(dims)}')
print(f'pauliZ(0) + pauliZ(1): {op8.concretize(dims)}')
print(f'pauliX(0) + pauliX(1) + pauliZ(0) + pauliZ(1): {(op7 + op8).concretize(dims)}')
print(f'(pauliX(0) + pauliX(1)) * (pauliZ(0) + pauliZ(1)): {(op7 * op8).concretize(dims)}')

print(f'pauliX(0) * (pauliZ(0) + pauliZ(1)): {(pauli.x(0) * op8).concretize(dims)}')
print(f'(pauliZ(0) + pauliZ(1)) * pauliX(0): {(op8 * pauli.x(0)).concretize(dims)}')

op9 = pauli.z(1) + pauli.z(2)
print(f'(pauliX(0) + pauliX(1)) * pauliI(2): {numpy.kron(op7.concretize(dims), pauli.i(2).concretize(dims))}')
print(f'(pauliX(0) + pauliX(1)) * pauliI(2): {(op7 * pauli.i(2)).concretize(dims)}')
print(f'(pauliX(0) + pauliX(1)) * pauliI(2): {(pauli.i(2) * op7).concretize(dims)}')
print(f'pauliI(0) * (pauliZ(1) + pauliZ(2)): {numpy.kron(pauli.i(0).concretize(dims), op9.concretize(dims))}')
print(f'(pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)): {(op7 * op9).concretize(dims)}')

so0 = ScalarOperator(lambda: 1.0j)
print(f'Scalar op (t -> 1.0)(): {so0.concretize()}')

so1 = ScalarOperator(lambda t: t)
print(f'Scalar op (t -> t)(1.): {so1.concretize(t = 1.0)}')
print(f'Trivial prod op (t -> t)(1.): {(ProductOperator([so1])).concretize({}, t = 1.)}')
print(f'Trivial prod op (t -> t)(2.): {(ProductOperator([so1])).concretize({}, t = 2.)}')

print(f'(t -> t)(1j) * pauliX(0): {(so1 * pauli.x(0)).concretize(dims, t = 1j)}')
print(f'pauliX(0) * (t -> t)(1j): {(pauli.x(0) * so1).concretize(dims, t = 1j)}')
print(f'pauliX(0) + (t -> t)(1j): {(pauli.x(0) + so1).concretize(dims, t = 1j)}')
print(f'(t -> t)(1j) + pauliX(0): {(so1 + pauli.x(0)).concretize(dims, t = 1j)}')
print(f'pauliX(0) + (t -> t)(1j): {(pauli.x(0) + so1).concretize(dims, t = 1j)}')
print(f'(t -> t)(1j) + pauliX(0): {(so1 + pauli.x(0)).concretize(dims, t = 1j)}')
print(f'(t -> t)(2.) * (pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)): {(so1 * op7 * op9).concretize(dims, t = 2.)}')
print(f'(pauliX(0) + pauliX(1)) * (t -> t)(2.) * (pauliZ(1) + pauliZ(2)): {(op7 * so1 * op9).concretize(dims, t = 2.)}')
print(f'(pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)) * (t -> t)(2.): {(op7 * op9 * so1).concretize(dims, t = 2.)}')

op10 = so1 * pauli.x(0)
so1.generator = lambda t: 1./t
print(f'(t -> 1/t)(2) * pauliX(0): {op10.concretize(dims, t = 2.)}')
so1_gen2 = so1.generator
so1.generator = lambda t: so1_gen2(2*t)
print(f'(t -> 1/(2t))(2) * pauliX(0): {op10.concretize(dims, t = 2.)}')
so1.generator = lambda t: so1_gen2(t)
print(f'(t -> 1/t)(2) * pauliX(0): {op10.concretize(dims, t = 2.)}')

so2 = ScalarOperator(lambda t: t**2)
op11 = pauli.z(1) * so2
print(f'pauliZ(0) * (t -> t^2)(2.): {op11.concretize(dims, t = 2.)}')

so3 = ScalarOperator(lambda t: 1./t)
so4 = ScalarOperator(lambda t: t**2)
print(f'((t -> 1/t) * (t -> t^2))(2.): {(so3 * so4).concretize(t = 2.)}')
so5 = so3 + so4
so3.generator = lambda field: 1./field
print(f'((f -> 1/f) + (t -> t^2))(f=2, t=1.): {so5.concretize(t = 1., field = 2)}')

def generator(field, **kwargs):
    print(f'generator got kwargs: {kwargs}')
    return field

so3.generator = generator
print(f'((f -> f) + (t -> t^2))(f=3, t=2): {so5.concretize(field = 3, t = 2, dummy = 10)}')

so6 = ScalarOperator(lambda foo, *, bar: foo * bar)
print(f'((f,t) -> f*t)(f=3, t=2): {so6.concretize(foo = 3, bar = 2, dummy = 10)}')
so7 = ScalarOperator(lambda foo, *, bar, **kwargs: foo * bar)
print(f'((f,t) -> f*t)(f=3, t=2): {so6.concretize(foo = 3, bar = 2, dummy = 10)}')

def get_parameter_value(parameter_name: str, time: float):
    match parameter_name:
        case "foo": return time
        case "bar": return 2 * time
        case _: raise NotImplementedError(f'No value defined for parameter {parameter_name}.')

schedule = Schedule([0.0, 0.5, 1.0], so6.parameters, get_parameter_value)
for parameters in schedule:
    print(f'step {schedule.current_step}')
    print(f'((f,t) -> f*t)({parameters}): {so6.concretize(**parameters)}')

print(f'(pauliX(0) + i*pauliY(0))/2: {0.5 * (pauli.x(0) + operators.const(1j) * pauli.y(0)).concretize(dims)}')
print(f'pauli+(0): {pauli.plus(0).concretize(dims)}')
print(f'(pauliX(0) - i*pauliY(0))/2: {0.5 * (pauli.x(0) - operators.const(1j) * pauli.y(0)).concretize(dims)}')
print(f'pauli-(0): {pauli.minus(0).concretize(dims)}')

op12 = operators.squeeze(0) + operators.displace(0)
print(f'create<3>(0): {operators.create(0).concretize({0:3})}')
print(f'annihilate<3>(0): {operators.annihilate(0).concretize({0:3})}')
print(f'squeeze<3>(0)[squeezing = 0.5]: {operators.squeeze(0).concretize({0:3}, squeezing=0.5)}')
print(f'displace<3>(0)[displacement = 0.5]: {operators.displace(0).concretize({0:3}, displacement=0.5)}')
print(f'(squeeze<3>(0) + displace<3>(0))[squeezing = 0.5, displacement = 0.5]: {op12.concretize({0:3}, displacement=0.5, squeezing=0.5)}')
print(f'squeeze<4>(0)[squeezing = 0.5]: {operators.squeeze(0).concretize({0:4}, squeezing=0.5)}')
print(f'displace<4>(0)[displacement = 0.5]: {operators.displace(0).concretize({0:4}, displacement=0.5)}')
print(f'(squeeze<4>(0) + displace<4>(0))[squeezing = 0.5, displacement = 0.5]: {op12.concretize({0:4}, displacement=0.5, squeezing=0.5)}')

so8 = ScalarOperator(lambda my_param: my_param - 1)
so9 = so7 * so8
print(f'parameter descriptions: {operators.squeeze(0).parameters}')
print(f'parameter descriptions: {op12.parameters}')
print(f'parameter descriptions: {(so7 + so8).parameters}')
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

#operators.const(1) * 5

scop = operators.const(1)
elop = operators.identity(1)
print(f"arithmetics: {scop.concretize(dims)}")
print(f"arithmetics: {elop.concretize(dims)}")
print(f"arithmetics: {(scop * elop).concretize(dims)}")
print(f"arithmetics: {(elop * scop).concretize(dims)}")
print(f"arithmetics: {(scop + elop).concretize(dims)}")
print(f"arithmetics: {(elop + scop).concretize(dims)}")
print(f"arithmetics: {(scop - elop).concretize(dims)}")
print(f"arithmetics: {(elop - scop).concretize(dims)}")
print(f"arithmetics: {((scop * elop) * scop).concretize(dims)}")
print(f"arithmetics: {(scop * (scop * elop)).concretize(dims)}")
print(f"arithmetics: {((scop * elop) * elop).concretize(dims)}")
print(f"arithmetics: {(elop * (scop * elop)).concretize(dims)}")
print(f"arithmetics: {((scop * elop) + scop).concretize(dims)}")
print(f"arithmetics: {(scop + (scop * elop)).concretize(dims)}")
print(f"arithmetics: {((scop * elop) + elop).concretize(dims)}")
print(f"arithmetics: {(elop + (scop * elop)).concretize(dims)}")
print(f"arithmetics: {((scop * elop) - scop).concretize(dims)}")
print(f"arithmetics: {(scop - (scop * elop)).concretize(dims)}")
print(f"arithmetics: {((scop * elop) - elop).concretize(dims)}")
print(f"arithmetics: {(elop - (scop * elop)).concretize(dims)}")
print(f"arithmetics: {((scop + elop) * scop).concretize(dims)}")
print(f"arithmetics: {(scop * (scop + elop)).concretize(dims)}")
print(f"arithmetics: {((scop + elop) * elop).concretize(dims)}")
print(f"arithmetics: {(elop * (scop + elop)).concretize(dims)}")
print(f"arithmetics: {((scop - elop) * scop).concretize(dims)}")
print(f"arithmetics: {(scop * (scop - elop)).concretize(dims)}")
print(f"arithmetics: {((scop - elop) * elop).concretize(dims)}")
print(f"arithmetics: {(elop * (scop - elop)).concretize(dims)}")
print(f"arithmetics: {((scop + elop) + scop).concretize(dims)}")
print(f"arithmetics: {(scop + (scop + elop)).concretize(dims)}")
print(f"arithmetics: {((scop + elop) + elop).concretize(dims)}")
print(f"arithmetics: {(elop + (scop + elop)).concretize(dims)}")
print(f"arithmetics: {((scop - elop) - scop).concretize(dims)}")
print(f"arithmetics: {(scop - (scop - elop)).concretize(dims)}")
print(f"arithmetics: {((scop - elop) - elop).concretize(dims)}")
print(f"arithmetics: {(elop - (scop - elop)).concretize(dims)}")

opprod = operators.create(0) * operators.annihilate(0)
opsum = operators.create(0) + operators.annihilate(0)
print(f"arithmetics: {(scop * 2).concretize(dims)}")
print(f"arithmetics: {(scop * 2.5).concretize(dims)}")
print(f"arithmetics: {(scop * 2j).concretize(dims)}")
print(f"arithmetics: {(2 * scop).concretize(dims)}")
print(f"arithmetics: {(2.5 * scop).concretize(dims)}")
print(f"arithmetics: {(2j * scop).concretize(dims)}")
print(f"arithmetics: {(elop * 2).concretize(dims)}")
print(f"arithmetics: {(elop * 2.5).concretize(dims)}")
print(f"arithmetics: {(elop * 2j).concretize(dims)}")
print(f"arithmetics: {(2 * elop).concretize(dims)}")
print(f"arithmetics: {(2.5 * elop).concretize(dims)}")
print(f"arithmetics: {(2j * elop).concretize(dims)}")
print(f"arithmetics: {(opprod * 2).concretize(dims)}")
print(f"arithmetics: {(opprod * 2.5).concretize(dims)}")
print(f"arithmetics: {(opprod * 2j).concretize(dims)}")
print(f"arithmetics: {(2 * opprod).concretize(dims)}")
print(f"arithmetics: {(2.5 * opprod).concretize(dims)}")
print(f"arithmetics: {(2j * opprod).concretize(dims)}")
print(f"arithmetics: {(opsum * 2).concretize(dims)}")
print(f"arithmetics: {(opsum * 2.5).concretize(dims)}")
print(f"arithmetics: {(opsum * 2j).concretize(dims)}")
print(f"arithmetics: {(2 * opsum).concretize(dims)}")
print(f"arithmetics: {(2.5 * opsum).concretize(dims)}")
print(f"arithmetics: {(2j * opsum).concretize(dims)}")
