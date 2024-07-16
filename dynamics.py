import numpy
from numpy.typing import NDArray

class BuiltIns:
    ops = {
        "spin_x": numpy.array([[0,1],[1,0]]),
        "spin_y": numpy.array([[0,1j],[-1j,0]]),
        "spin_z": numpy.array([[1,0],[0,-1]]),
        "spin_i": numpy.array([[1,0],[0,1]]),
    }

    # From https://en.wikipedia.org/wiki/Pauli_matrices (Cayley table)
    # FIXME: missing constants
    pauli_table = [
        ["spin_i", "spin_z", "spin_y", "spin_x"],
        ["spin_z", "spin_i", "spin_x", "spin_y"],
        ["spin_y", "spin_x", "spin_i", "spin_z"],
        ["spin_x", "spin_y", "spin_z", "spin_i"],
    ]

    @staticmethod
    def pauli_id(arg: str):
        match arg:
            case "spin_x": return 0
            case "spin_y": return 1
            case "spin_z": return 2
            case "spin_i": return 3

    @staticmethod
    def product(arg1: str, arg2: str):
        if not arg1.startswith("spin_") or not arg2.startswith("spin_"):
            raise NotImplementedError("Only implemented spin operators so far.")
        row = BuiltIns.pauli_id(arg1)
        column = BuiltIns.pauli_id(arg2)
        return BuiltIns.pauli_table[row][column]


class SimpleOperator():
    pass

class ProductOperator():
    pass

class OperatorSum():
    pass

class OperatorSum:
    _terms: list[ProductOperator]

    def __init__(self, terms: list[ProductOperator]):
        if len(terms) == 0:
            raise ValueError("Need at least one term.")
        self._terms = terms

    def concretize(self, levels: dict[int, int]) -> NDArray[complex]:
        degrees = set([op._degree for term in self._terms for op in term._operators])
        padded_terms = []
        for term in self._terms:
            for degree in degrees:
                if not degree in [op._degree for op in term._operators]:
                    term *= spin.i(degree)
            padded_terms.append(term)
        return sum([term.concretize(levels) for term in padded_terms])

    def __mul__(self, other: OperatorSum):
        if type(other) == SimpleOperator:
            return self * OperatorSum([ProductOperator([other])])
        elif type(other) == ProductOperator:
            return self * OperatorSum([other])
        return OperatorSum([self_term * other_term for self_term in self._terms for other_term in other._terms])
    
    def __add__(self, other: OperatorSum):
        if type(other) == SimpleOperator:
            return self + OperatorSum([ProductOperator([other])])
        elif type(other) == ProductOperator:
            return self + OperatorSum([other])
        return OperatorSum(self._terms + other._terms)

class ProductOperator(OperatorSum):
    _operators: list[SimpleOperator]

    def __init__(self, operators : list[SimpleOperator]):
        if len(operators) == 0:
            raise ValueError("Need at least one operator.")
        operators.sort(key = lambda op: op._degree)
        for i in range(1, len(operators)):
            if operators[i]._degree == operators[i-1]._degree:
                raise ValueError("Multiple operators cannot act on the same degree of freedom.")
        self._operators = operators
        super().__init__([self])

    def concretize(self, levels: dict[int, int]) -> NDArray[complex]:
        matrix = self._operators[0].concretize(levels)
        for op in self._operators[1:]:
            matrix = numpy.kron(matrix, op.concretize(levels))
        return matrix

    def __mul__(self, other: ProductOperator):
        if type(other) == SimpleOperator:
            return self * ProductOperator([other])
        elif type(other) != ProductOperator:
            return OperatorSum([self]) * other

        self_idx, other_idx = 0, 0
        operators = []
        # If the two operators don't share a degree of freedom, 
        # we could just return ProductOperator(self._operators + other._operators).
        # In the case they share a degree of freedom, we need to take the product
        # for that degree since a product operator expects unique degrees.
        # This while loop just takes care of the sorting while we are at it.
        while self_idx < len(self._operators):
            while other_idx < len(other._operators) and other._operators[other_idx]._degree < self._operators[self_idx]._degree:
                operators += [other._operators[other_idx]]
                other_idx += 1
            if other_idx < len(other._operators) and other._operators[other_idx]._degree == self._operators[self_idx]._degree:
                operators += [self._operators[self_idx] * other._operators[other_idx]]
                other_idx += 1
            else: 
                operators += [self._operators[self_idx]]
            self_idx += 1
        for other_op in other._operators[other_idx:]:
            operators += [other_op]
        return ProductOperator(operators)

    def __add__(self, other: ProductOperator):
        if type(other) == SimpleOperator:
            return self + OperatorSum([other])
        elif type(other) != ProductOperator:
            return OperatorSum([self]) + other
        return OperatorSum([self, other])

class SimpleOperator(ProductOperator):
    _degree: int
    _builtin_id: str

    def __init__(self, degree: int, builtin_id: str):
        self._degree = degree
        self._builtin_id = builtin_id
        super().__init__([self])

    def concretize(self, levels: dict[int, int]) -> NDArray[complex]:
        if self._degree not in levels:
            raise ValueError(f'Missing levels for degree {self._degree}')

        if levels[self._degree] != 2:
           raise NotImplementedError() # FIXME

        return BuiltIns.ops[self._builtin_id]

    def __mul__(self, other: SimpleOperator):
        if type(other) != SimpleOperator:
            return ProductOperator([self]) * other

        if self._degree == other._degree:
            product_id = BuiltIns.product(self._builtin_id, other._builtin_id)
            return SimpleOperator(self._degree, product_id)
        else:
            return ProductOperator([self, other])

    def __add__(self, other: SimpleOperator):
        if type(other) != SimpleOperator:
            return ProductOperator([self]) + other
        op1 = ProductOperator([self])
        op2 = ProductOperator([other])
        return OperatorSum([op1, op2])

# Operators as defined here: 
# https://www.dynamiqs.org/python_api/utils/operators/sigmay.html

class spin:

    @classmethod
    def x(cls, degree: int) -> SimpleOperator:
        return SimpleOperator(degree, "spin_x")
    @classmethod
    def y(cls, degree: int) -> SimpleOperator:
        return SimpleOperator(degree, "spin_y")
    @classmethod
    def z(cls, degree: int) -> SimpleOperator:
        return SimpleOperator(degree, "spin_z")
    @classmethod
    def i(cls, degree: int) -> SimpleOperator:
        return SimpleOperator(degree, "spin_i")
    
levels = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}
print(f'spinX(1): {spin.x(1).concretize(levels)}')
print(f'spinY(2): {spin.y(2).concretize(levels)}')

print(f'spinZ(0) * spinZ(0): {(spin.z(0) * spin.z(0)).concretize(levels)}')
print(f'spinZ(0) * spinZ(1): {(spin.z(0) * spin.z(1)).concretize(levels)}')
print(f'spinZ(0) * spinY(1): {(spin.z(0) * spin.y(1)).concretize(levels)}')

op1 = ProductOperator([spin.x(0), spin.i(1)])
op2 = ProductOperator([spin.i(0), spin.x(1)])
print(f'spinX(0) + spinX(1): {op1.concretize(levels) + op2.concretize(levels)}')
op3 = ProductOperator([spin.x(1), spin.i(0)])
op4 = ProductOperator([spin.i(1), spin.x(0),])
print(f'spinX(1) + spinX(0): {op1.concretize(levels) + op2.concretize(levels)}')

print(f'spinX(0) + spinX(1): {(spin.x(0) + spin.x(1)).concretize(levels)}')

print(f'spinX(0) * spinX(1): {(spin.x(0) * spin.x(1)).concretize(levels)}')
print(f'spinX(0) * spinI(1) * spinI(0) * spinX(1): {(op1 * op2).concretize(levels)}')

print(f'spinX(0) * spinI(1): {op1.concretize(levels)}')
print(f'spinI(0) * spinX(1): {op2.concretize(levels)}')
print(f'spinX(0) * spinI(1) + spinI(0) * spinX(1): {(op1 + op2).concretize(levels)}')

op5 = spin.x(0) * spin.x(1)
op6 = spin.z(0) * spin.z(1)
print(f'spinX(0) * spinX(1): {op5.concretize(levels)}')
print(f'spinZ(0) * spinZ(1): {op6.concretize(levels)}')
print(f'spinX(0) * spinX(1) + spinZ(0) * spinZ(1): {(op5 + op6).concretize(levels)}')

op7 = spin.x(0) + spin.x(1)
op8 = spin.z(0) + spin.z(1)
print(f'spinX(0) + spinX(1): {op7.concretize(levels)}')
print(f'spinZ(0) + spinZ(1): {op8.concretize(levels)}')
print(f'spinX(0) + spinX(1) + spinZ(0) + spinZ(1): {(op7 + op8).concretize(levels)}')
print(f'(spinX(0) + spinX(1)) * (spinZ(0) + spinZ(1)): {(op7 * op8).concretize(levels)}')

print(f'spinX(0) * (spinZ(0) + spinZ(1)): {(spin.x(0) * op8).concretize(levels)}')
print(f'(spinZ(0) + spinZ(1)) * spinX(0): {(op8 * spin.x(0)).concretize(levels)}')
