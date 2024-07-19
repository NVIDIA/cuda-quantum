import inspect, itertools, numpy, scipy
from numbers import Number
from typing import Any, Callable, Iterable, Iterator
from numpy.typing import NDArray


class ScalarOperator():
    pass

class ElementaryOperator():
    pass

class ProductOperator():
    pass

class OperatorSum():
    pass

class OperatorSum:

    def __init__(self, terms: list[ProductOperator]):
        if len(terms) == 0:
            raise ValueError("need at least one term")
        self._terms = terms

    @property
    def parameter_names(self) -> set[str]:
        return set([name for term in self._terms for name in term.parameter_names])

    def concretize(self, levels: dict[int, int], **kwargs) -> NDArray[complex]:
        degrees = set([degree for term in self._terms for op in term._operators for degree in op._degrees])
        padded_terms = [] # We need to make sure all matrices are of the same size to sum them up.
        for term in self._terms:
            for degree in degrees:
                if not degree in [degree for op in term._operators for degree in op._degrees]:
                    term *= operators.identity(degree)
            padded_terms.append(term)
        return sum([term.concretize(levels, **kwargs) for term in padded_terms])

    def __mul__(self, other: OperatorSum):
        if not isinstance(other, OperatorSum):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")
        elif type(other) == ScalarOperator:
            return self * OperatorSum([ProductOperator([other])])
        elif type(other) == ElementaryOperator:
            return self * OperatorSum([ProductOperator([other])])
        elif type(other) == ProductOperator:
            return self * OperatorSum([other])
        return OperatorSum([self_term * other_term for self_term in self._terms for other_term in other._terms])
    
    def __add__(self, other: OperatorSum):
        if not isinstance(other, OperatorSum):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")
        elif type(other) == ScalarOperator:
            return self + OperatorSum([ProductOperator([other])])
        elif type(other) == ElementaryOperator:
            return self + OperatorSum([ProductOperator([other])])
        elif type(other) == ProductOperator:
            return self + OperatorSum([other])
        return OperatorSum(self._terms + other._terms)

class ProductOperator(OperatorSum):

    def __init__(self, operators : list[ElementaryOperator]):
        if len(operators) == 0:
            raise ValueError("need at least one operator")
        self._operators = operators
        super().__init__([self])

    @property
    def parameter_names(self) -> set[str]:
        return set([name for operator in self._operators for name in operator.parameter_names])

    def concretize(self, levels: dict[int, int], **kwargs) -> NDArray[complex]:
        def generate_all_states(degrees: list[int]):
            if len(degrees) == 0:
                return []
            states = [[str(state)] for state in range(levels[degrees[0]])]
            for d in degrees[1:]:
                prod = itertools.product(states, [str(state) for state in range(levels[d])])
                states = [current + [new] for current, new in prod]
            return [''.join(state) for state in states]

        def padded_matrix(op: ElementaryOperator, degrees: list[int]):
            op_matrix = op.concretize(levels, **kwargs)
            op_degrees = op._degrees.copy() # Determines the initial qubit ordering of op_matrix.
            for degree in degrees:
                if not degree in [d for d in op._degrees]:
                    op_matrix = numpy.kron(op_matrix, operators.identity(degree).concretize(levels, **kwargs))
                    op_degrees.append(degree)
            # Need to permute the matrix such that the qubit ordering of all matrices is the same.
            if op_degrees != degrees:
                # I'm sure there is a more efficient way, but needed something correct first.
                states = generate_all_states(degrees) # ordered according to degrees
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
        # FIXME: make the overall degrees of the product op accessible?
        # FIXME: check endianness ...
        matrix = padded_matrix(self._operators[0], degrees)
        for op in self._operators[1:]:
            matrix = numpy.dot(matrix, padded_matrix(op, degrees))
        return matrix

    def __mul__(self, other: ProductOperator):
        if not isinstance(other, OperatorSum):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")
        elif type(other) == ScalarOperator:
            return self * ProductOperator([other])
        elif type(other) == ElementaryOperator:
            return self * ProductOperator([other])
        elif type(other) != ProductOperator:
            return OperatorSum([self]) * other
        return ProductOperator(self._operators + other._operators)

    def __add__(self, other: ProductOperator):
        if not isinstance(other, OperatorSum):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")
        elif type(other) == ScalarOperator:
            return self + ProductOperator([other])
        elif type(other) == ElementaryOperator:
            return self + OperatorSum([other])
        elif type(other) != ProductOperator:
            return OperatorSum([self]) + other
        return OperatorSum([self, other])

    def __sub__(self, other: ScalarOperator) -> ScalarOperator:
        return self + operators.const(-1) * other

class ElementaryOperator(ProductOperator):
    _ops = {}

    # The Callable `create` that is passed here may take any number and types of 
    # arguments and must return a NDArray[complex]. Each argument must be passed
    # as keyword arguments when concretizing any operator that involves this built-in.
    # Note that the levels passed to the create function are automatically validated 
    # against the expected/supported levels passed to `add_operator`. There is hence 
    # no need to validate the levels as part of the `create` function. 
    # A negative or zero value for one (or more) of the expected level indicates that 
    # the matrix/operator is defined for any value of this level. If the operator
    # definition depends on the level(s), then the `create` function must take an 
    # argument called `levels`, or `level` if it is just one. The given list of levels,
    # or its only entry if it is a list of length one, will then be automatically 
    # forwarded as argument to `create`.
    @classmethod
    def define(cls, op_id: str, expected_levels: list[int], create: Callable):
        def with_level_check(generator, levels: list[int], **kwargs) -> Callable:
            # Passing a value 0 for one of the expected levels indicates that
            # the generator can be invoked with any value for that level.
            # The generator returns a function that, given some keyword arguments,
            # returns a matrix (NDArray[complex]).
            if any([expected > 0 and levels[i] != expected for i, expected in enumerate(expected_levels)]):
                raise ValueError(f'no built-in operator {op_id} has been defined '\
                                 f'for {len(levels)} degree(s) of freedom with level(s) {levels}')
            if len(levels) == 1: kwargs["level"] = levels[0]
            else: kwargs["levels"] = levels 
            generator_args, remaining_kwargs = ScalarOperator._args_from_kwargs(generator, **kwargs)
            evaluated = generator(*generator_args, **remaining_kwargs)
            if not isinstance(evaluated, numpy.ndarray):
                raise TypeError("operator concretization must return a 'NDArray[complex]'")
            return evaluated
        cls._ops[op_id] = lambda levels, **kwargs: with_level_check(create, levels, **kwargs)

    def __init__(self, operator_id: str, degrees: list[int]):
        if not operator_id in ElementaryOperator._ops:
            raise ValueError(f"no built-in operator '{operator_id}' has been defined")
        self._generator = ElementaryOperator._ops[operator_id]
        self._degrees = degrees
        self._degrees.sort() # sorting so that we have a unique ordering for builtin
        super().__init__([self])

    @property
    def parameter_names(self) -> set[str]:
        arg_spec = inspect.getfullargspec(self._generator)
        return set([arg_name for arg_name in arg_spec.args + arg_spec.kwonlyargs])

    def concretize(self, levels: dict[int, int], **kwargs) -> NDArray[complex]:
        missing_degrees = [degree not in levels for degree in self._degrees]
        if any(missing_degrees):
            raise ValueError(f'missing levels for degree(s) {[self._degrees[i] for i, x in enumerate(missing_degrees) if x]}')
        relevant_levels = [levels[degree] for degree in self._degrees]
        return self._generator(relevant_levels, **kwargs)

    def __mul__(self, other: ElementaryOperator):
        if not isinstance(other, OperatorSum):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")
        elif type(other) == ScalarOperator:
            return ProductOperator([self]) * ProductOperator([other])
        elif type(other) != ElementaryOperator:
            return ProductOperator([self]) * other
        return ProductOperator([self, other])

    def __add__(self, other: ElementaryOperator):
        if not isinstance(other, OperatorSum):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")
        elif type(other) == ScalarOperator:
            return ProductOperator([self]) + ProductOperator([other])
        elif type(other) != ElementaryOperator:
            return ProductOperator([self]) + other
        op1 = ProductOperator([self])
        op2 = ProductOperator([other])
        return OperatorSum([op1, op2])

    def __sub__(self, other: ScalarOperator) -> ScalarOperator:
        return self + operators.const(-1) * other

class ScalarOperator(ProductOperator):

    # The given generator may take any number and types of arguments, 
    # and must return a numeric value. Each argument must be passed
    # as keyword arguments when concretizing any operator that involves
    # this scalar operator.
    def __init__(self, generator: Callable):
        self._degrees = []
        self.generator = generator # Don't set self._generator directly; the setter validates the value.
        super().__init__([self])

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, generator: Callable):
        # A variable number of arguments (i.e. *args) cannot be supported
        # for generators; it would prevent proper argument handling while 
        # supporting additions and multiplication of all kinds of operators.
        arg_spec = inspect.getfullargspec(generator)
        if arg_spec.varargs is not None:
            raise ValueError(f"generator for a '{type(self).__name__}' must not take *args")
        self._generator = generator

    @property
    def parameter_names(self) -> set[str]:
        arg_spec = inspect.getfullargspec(self._generator)
        return set([arg_name for arg_name in arg_spec.args + arg_spec.kwonlyargs])

    def _args_from_kwargs(fct, **kwargs):
        arg_spec = inspect.getfullargspec(fct)
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
                default_value = inspect.signature(fct).parameters[arg_name].default
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

    # The argument `levels` here is only passed for consistency with parent classes.
    def concretize(self, levels: dict[int, int] = None, **kwargs):
        generator_args, remaining_kwargs = ScalarOperator._args_from_kwargs(self._generator, **kwargs)
        evaluated = self._generator(*generator_args, **remaining_kwargs)
        if not isinstance(evaluated, Number):
            raise TypeError(f"generator of {type(self).__name__} must return a 'Number'")
        return evaluated

    def __mul__(self, other: ScalarOperator) -> ScalarOperator:
        if not isinstance(other, OperatorSum):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")
        elif type(other) != ScalarOperator:
            return ProductOperator([self]) * other
        generator = lambda **kwargs: self.concretize(**kwargs) * other.concretize(**kwargs)
        return ScalarOperator(generator)
    
    def __add__(self, other: ScalarOperator) -> ScalarOperator:
        if not isinstance(other, OperatorSum):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other).__name__}'")
        elif type(other) != ScalarOperator:
            return ProductOperator([self]) + other
        generator = lambda **kwargs: self.concretize(**kwargs) + other.concretize(**kwargs)
        return ScalarOperator(generator)

    def __sub__(self, other: ScalarOperator) -> ScalarOperator:
        return self + operators.const(-1) * other


# Operators as defined here: 
# https://www.dynamiqs.org/python_api/utils/operators/sigmay.html
class operators:

    def _create(level: int): 
        return numpy.diag(numpy.sqrt(numpy.arange(1, level, dtype=complex)), -1)
    def _annihilate(level: int): 
        return numpy.diag(numpy.sqrt(numpy.arange(1, level, dtype=complex)), 1)
    def _position(level: int):
        return 0.5 * (operators._create(level) + operators._annihilate(level))
    def _momentum(level: int):
        return 0.5j * (operators._create(level) - operators._annihilate(level))
    def _displace(level: int, displacement: complex):
        term1 = displacement * operators._create(level)
        term2 = numpy.conjugate(displacement) * operators._annihilate(level)
        return scipy.linalg.expm(term1 - term2)
    def _squeeze(level: int, squeezing: complex):
        term1 = numpy.conjugate(squeezing) * numpy.linalg.matrix_power(operators._annihilate(level), 2)
        term2 = squeezing * numpy.linalg.matrix_power(operators._create(level), 2)
        return scipy.linalg.expm(0.5 * (term1 - term2))

    ElementaryOperator.define("op_zero", [0], lambda level: numpy.zeros((level, level), dtype=complex))
    ElementaryOperator.define("op_identity", [0], lambda level: numpy.diag(numpy.ones(level, dtype=complex)))
    ElementaryOperator.define("op_create", [0], _create)
    ElementaryOperator.define("op_annihilate", [0], _annihilate)
    ElementaryOperator.define("op_number", [0], lambda level: numpy.diag(numpy.arange(level, dtype=complex)))
    ElementaryOperator.define("op_parity", [0], lambda level: numpy.diag([(-1.+0j)**i for i in range(level)]))
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
    _parameters: list[str]
    _get_value: Callable[[str, Any], Any]
    _current_step: Any

    # The output type of the iterable steps must match the second argument of `get_value`.
    def __init__(self, steps: Iterable[Any], parameters: list[str], get_value: Callable[[str, Any], Any]):
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


levels = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}

print(f'pauliX(1): {pauli.x(1).concretize(levels)}')
print(f'pauliY(2): {pauli.y(2).concretize(levels)}')

print(f'pauliZ(0) * pauliZ(0): {(pauli.z(0) * pauli.z(0)).concretize(levels)}')
print(f'pauliZ(0) * pauliZ(1): {(pauli.z(0) * pauli.z(1)).concretize(levels)}')
print(f'pauliZ(0) * pauliY(1): {(pauli.z(0) * pauli.y(1)).concretize(levels)}')

op1 = ProductOperator([pauli.x(0), pauli.i(1)])
op2 = ProductOperator([pauli.i(0), pauli.x(1)])
print(f'pauliX(0) + pauliX(1): {op1.concretize(levels) + op2.concretize(levels)}')
op3 = ProductOperator([pauli.x(1), pauli.i(0)])
op4 = ProductOperator([pauli.i(1), pauli.x(0),])
print(f'pauliX(1) + pauliX(0): {op1.concretize(levels) + op2.concretize(levels)}')

print(f'pauliX(0) + pauliX(1): {(pauli.x(0) + pauli.x(1)).concretize(levels)}')
print(f'pauliX(0) * pauliX(1): {(pauli.x(0) * pauli.x(1)).concretize(levels)}')
print(f'pauliX(0) * pauliI(1) * pauliI(0) * pauliX(1): {(op1 * op2).concretize(levels)}')

print(f'pauliX(0) * pauliI(1): {op1.concretize(levels)}')
print(f'pauliI(0) * pauliX(1): {op2.concretize(levels)}')
print(f'pauliX(0) * pauliI(1) + pauliI(0) * pauliX(1): {(op1 + op2).concretize(levels)}')

op5 = pauli.x(0) * pauli.x(1)
op6 = pauli.z(0) * pauli.z(1)
print(f'pauliX(0) * pauliX(1): {op5.concretize(levels)}')
print(f'pauliZ(0) * pauliZ(1): {op6.concretize(levels)}')
print(f'pauliX(0) * pauliX(1) + pauliZ(0) * pauliZ(1): {(op5 + op6).concretize(levels)}')

op7 = pauli.x(0) + pauli.x(1)
op8 = pauli.z(0) + pauli.z(1)
print(f'pauliX(0) + pauliX(1): {op7.concretize(levels)}')
print(f'pauliZ(0) + pauliZ(1): {op8.concretize(levels)}')
print(f'pauliX(0) + pauliX(1) + pauliZ(0) + pauliZ(1): {(op7 + op8).concretize(levels)}')
print(f'(pauliX(0) + pauliX(1)) * (pauliZ(0) + pauliZ(1)): {(op7 * op8).concretize(levels)}')

print(f'pauliX(0) * (pauliZ(0) + pauliZ(1)): {(pauli.x(0) * op8).concretize(levels)}')
print(f'(pauliZ(0) + pauliZ(1)) * pauliX(0): {(op8 * pauli.x(0)).concretize(levels)}')

op9 = pauli.z(1) + pauli.z(2)
print(f'(pauliX(0) + pauliX(1)) * pauliI(2): {numpy.kron(op7.concretize(levels), pauli.i(2).concretize(levels))}')
print(f'(pauliX(0) + pauliX(1)) * pauliI(2): {(op7 * pauli.i(2)).concretize(levels)}')
print(f'(pauliX(0) + pauliX(1)) * pauliI(2): {(pauli.i(2) * op7).concretize(levels)}')
print(f'pauliI(0) * (pauliZ(1) + pauliZ(2)): {numpy.kron(pauli.i(0).concretize(levels), op9.concretize(levels))}')
print(f'(pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)): {(op7 * op9).concretize(levels)}')

so0 = ScalarOperator(lambda: 1.0j)
print(f'Scalar op (t -> 1.0)(): {so0.concretize()}')

so1 = ScalarOperator(lambda t: t)
print(f'Scalar op (t -> t)(1.): {so1.concretize(t = 1.0)}')
print(f'Trivial prod op (t -> t)(1.): {(ProductOperator([so1])).concretize({}, t = 1.)}')
print(f'Trivial prod op (t -> t)(2.): {(ProductOperator([so1])).concretize({}, t = 2.)}')

print(f'(t -> t)(1j) * pauliX(0): {(so1 * pauli.x(0)).concretize(levels, t = 1j)}')
print(f'pauliX(0) * (t -> t)(1j): {(pauli.x(0) * so1).concretize(levels, t = 1j)}')
print(f'pauliX(0) + (t -> t)(1j): {(pauli.x(0) + so1).concretize(levels, t = 1j)}')
print(f'(t -> t)(1j) + pauliX(0): {(so1 + pauli.x(0)).concretize(levels, t = 1j)}')
print(f'pauliX(0) + (t -> t)(1j): {(pauli.x(0) + so1).concretize(levels, t = 1j)}')
print(f'(t -> t)(1j) + pauliX(0): {(so1 + pauli.x(0)).concretize(levels, t = 1j)}')
print(f'(t -> t)(2.) * (pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)): {(so1 * op7 * op9).concretize(levels, t = 2.)}')
print(f'(pauliX(0) + pauliX(1)) * (t -> t)(2.) * (pauliZ(1) + pauliZ(2)): {(op7 * so1 * op9).concretize(levels, t = 2.)}')
print(f'(pauliX(0) + pauliX(1)) * (pauliZ(1) + pauliZ(2)) * (t -> t)(2.): {(op7 * op9 * so1).concretize(levels, t = 2.)}')

op10 = so1 * pauli.x(0)
so1.generator = lambda t: 1./t
print(f'(t -> 1/t)(2) * pauliX(0): {op10.concretize(levels, t = 2.)}')
so1_gen2 = so1.generator
so1.generator = lambda t: so1_gen2(2*t)
print(f'(t -> 1/(2t))(2) * pauliX(0): {op10.concretize(levels, t = 2.)}')
so1.generator = lambda t: so1_gen2(t)
print(f'(t -> 1/t)(2) * pauliX(0): {op10.concretize(levels, t = 2.)}')

so2 = ScalarOperator(lambda t: t**2)
op11 = pauli.z(1) * so2
print(f'pauliZ(0) * (t -> t^2)(2.): {op11.concretize(levels, t = 2.)}')

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

schedule = Schedule([0.0, 0.5, 1.0], so6.parameter_names, get_parameter_value)
for parameters in schedule:
    print(f'step {schedule.current_step}')
    print(f'((f,t) -> f*t)({parameters}): {so6.concretize(**parameters)}')

print(f'(pauliX(0) + i*pauliY(0))/2: {0.5 * (pauli.x(0) + operators.const(1j) * pauli.y(0)).concretize(levels)}')
print(f'pauli+(0): {pauli.plus(0).concretize(levels)}')
print(f'(pauliX(0) - i*pauliY(0))/2: {0.5 * (pauli.x(0) - operators.const(1j) * pauli.y(0)).concretize(levels)}')
print(f'pauli-(0): {pauli.minus(0).concretize(levels)}')

op12 = operators.squeeze(0) + operators.displace(0)
print(f'create<3>(0): {operators.create(0).concretize({0:3})}')
print(f'annihilate<3>(0): {operators.annihilate(0).concretize({0:3})}')
print(f'squeeze<3>(0)[squeezing = 0.5]: {operators.squeeze(0).concretize({0:3}, squeezing=0.5)}')
print(f'displace<3>(0)[displacement = 0.5]: {operators.displace(0).concretize({0:3}, displacement=0.5)}')
print(f'(squeeze<3>(0) + displace<3>(0))[squeezing = 0.5, displacement = 0.5]: {op12.concretize({0:3}, displacement=0.5, squeezing=0.5)}')
print(f'squeeze<4>(0)[squeezing = 0.5]: {operators.squeeze(0).concretize({0:4}, squeezing=0.5)}')
print(f'displace<4>(0)[displacement = 0.5]: {operators.displace(0).concretize({0:4}, displacement=0.5)}')
print(f'(squeeze<4>(0) + displace<4>(0))[squeezing = 0.5, displacement = 0.5]: {op12.concretize({0:4}, displacement=0.5, squeezing=0.5)}')
