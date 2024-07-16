import numpy
from numpy.typing import NDArray

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
        degrees = [op._degree for op in terms[0]._operators]
        for term in terms[1:]:
            if degrees != [op._degree for op in term._operators]:
                raise ValueError("All operators must act on the same degrees of freedom")
        self._terms = terms

    def _add_identity_padding(prod1: ProductOperator, prod2: ProductOperator):
        prod1_ops, prod2_ops = [], []
        prod1_idx, prod2_idx = 0, 0
        while prod1_idx < len(prod1._operators):
            while prod2_idx < len(prod2._operators) and prod2._operators[prod2_idx]._degree < prod1._operators[prod1_idx]._degree:
                prod1_ops.append(spin.i(prod2._operators[prod2_idx]._degree))
                prod2_ops.append(prod2._operators[prod2_idx])
                prod2_idx += 1
            if prod2._operators[prod2_idx]._degree == prod1._operators[prod1_idx]._degree:
                prod1_ops.append(prod1._operators[prod1_idx])
                prod2_ops.append(prod2._operators[prod2_idx])
                prod2_idx += 1
            else: 
                prod1_ops.append(prod1._operators[prod1_idx])
                prod2_ops.append(spin.i(prod1._operators[prod1_idx]._degree))
            prod1_idx += 1
        for other_op in prod2._operators[prod2_idx:]:
            prod2_ops.append(other_op)
            prod1_ops.append(spin.i(other_op._degree))
        return ProductOperator(prod1_ops), ProductOperator(prod2_ops)

    @property
    def matrix(self) -> NDArray[complex]:
        matrix = self._terms[0].matrix
        for term in self._terms[1:]:
            matrix += term.matrix
        return matrix
    
    def add_product(self, prod: ProductOperator) -> OperatorSum:
        # If prod acts on different degrees of freedom than self, we add 
        # a product with the identity for the degree that is not represented in 
        # some of the product operators.
        # FIXME: we could probably just do this when we concretize the matrix,
        # instead of upon construction of the operator sum.
        padded_self, padded_prod = OperatorSum._add_identity_padding(self._terms[0], prod)
        if len(padded_self._operators) == len(self._terms[0]._operators):
            terms = self._terms
        else:
            terms = [padded_self]
            for term in self._terms[1:]:
                terms.append(OperatorSum._add_identity_padding(padded_self, term)[1])
        terms.append(padded_prod)
        return OperatorSum(terms)

    def __mul__(self, other: OperatorSum):
        if type(other) == SimpleOperator:
            return self * OperatorSum([ProductOperator([other])])
        elif type(other) == ProductOperator:
            return self * OperatorSum([other])

        terms = []
        for self_term in self._terms:
            for other_term in other._terms:
                terms.append(self_term * other_term)
        return OperatorSum(terms)
    
    def __add__(self, other: OperatorSum):
        if type(other) == SimpleOperator:
            return self + OperatorSum([ProductOperator([other])])
        elif type(other) == ProductOperator:
            return self + OperatorSum([other])

        # Make sure all operators act on the same degrees of freedom.
        sum = self.add_product(other._terms[0])
        if len(sum._terms[0]._operators) == len(other._terms[0]._operators):
            return OperatorSum(sum._terms + other._terms[1:])
        else:
            for term in other._terms[1:]:
                sum = sum.add_product(term)
            return sum

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

    @property
    def matrix(self) -> NDArray[complex]:
        matrix = self._operators[0].matrix
        for op in self._operators[1:]:
            matrix = numpy.kron(matrix, op.matrix)
        return matrix

    def __mul__(self, other: ProductOperator):
        if type(other) == SimpleOperator:
            return self * OperatorSum([other])
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
        return OperatorSum([self]).add_product(other)

class SimpleOperator(ProductOperator):
    _degree: int
    _matrix: NDArray[complex]

    def __init__(self, degree: int, matrix: NDArray[complex]):
        self._degree = degree
        self._matrix = matrix
        super().__init__([self])

    @property
    def matrix(self) -> NDArray[complex]:
       return self._matrix

    def __mul__(self, other: SimpleOperator):
        if type(other) != SimpleOperator:
            return ProductOperator([self]) * other

        if self._degree == other._degree:
            matrix = numpy.dot(self.matrix, other.matrix) # FIXME: replace with symbolic
            return SimpleOperator(self._degree, matrix)
        else:
            return ProductOperator([self, other])

    def __add__(self, other: SimpleOperator):
        if type(other) != SimpleOperator:
            return ProductOperator([self]) + other

        if self._degree == other._degree:
            matrix = self.matrix + other.matrix
            return SimpleOperator(self._degree, matrix)
        else:
            op1 = ProductOperator([self, spin.i(other._degree)])
            op2 = ProductOperator([spin.i(self._degree), other])
            return OperatorSum([op1, op2])

# Operators as defined here: 
# https://www.dynamiqs.org/python_api/utils/operators/sigmay.html

class spin: # FIXME: matrix only valid when on qubits... -> concretized...

    @classmethod
    def x(cls, degree: int) -> SimpleOperator:
        return SimpleOperator(degree, numpy.array([[0,1],[1,0]]))
    @classmethod
    def y(cls, degree: int) -> SimpleOperator:
        return SimpleOperator(degree, numpy.array([[0,1j],[-1j,0]]))
    @classmethod
    def z(cls, degree: int) -> SimpleOperator:
        return SimpleOperator(degree, numpy.array([[1,0],[0,-1]]))
    @classmethod
    def i(cls, degree: int) -> SimpleOperator:
        return SimpleOperator(degree, numpy.array([[1,0],[0,1]]))
    

print(f'spinX(1): {spin.x(1).matrix}')
print(f'spinY(2): {spin.y(2).matrix}')

print(f'spinZ(0) * spinZ(0): {(spin.z(0) * spin.z(0)).matrix}')
print(f'spinZ(0) * spinZ(1): {(spin.z(0) * spin.z(1)).matrix}')
print(f'spinZ(0) * spinY(1): {(spin.z(0) * spin.y(1)).matrix}')

op1 = ProductOperator([spin.x(0), spin.i(1)])
op2 = ProductOperator([spin.i(0), spin.x(1)])
print(f'spinX(0) + spinX(1): {op1.matrix + op2.matrix}')
op3 = ProductOperator([spin.x(1), spin.i(0)])
op4 = ProductOperator([spin.i(1), spin.x(0),])
print(f'spinX(1) + spinX(0): {op1.matrix + op2.matrix}')

print(f'spinX(0) + spinX(1): {(spin.x(0) + spin.x(1)).matrix}')

print(f'spinX(0) * spinX(1): {(spin.x(0) * spin.x(1)).matrix}')
print(f'spinX(0) * spinI(1) * spinI(0) * spinX(1): {(op1 * op2).matrix}')

print(f'spinX(0) * spinI(1): {op1.matrix}')
print(f'spinI(0) * spinX(1): {op2.matrix}')
print(f'spinX(0) * spinI(1) + spinI(0) * spinX(1): {(op1 + op2).matrix}')

op5 = spin.x(0) * spin.x(1)
op6 = spin.z(0) * spin.z(1)
print(f'spinX(0) * spinX(1): {op5.matrix}')
print(f'spinZ(0) * spinZ(1): {op6.matrix}')
print(f'spinX(0) * spinX(1) + spinZ(0) * spinZ(1): {(op5 + op6).matrix}')

op7 = spin.x(0) + spin.x(1)
op8 = spin.z(0) + spin.z(1)
print(f'spinX(0) + spinX(1): {op7.matrix}')
print(f'spinZ(0) + spinZ(1): {op8.matrix}')
print(f'spinX(0) + spinX(1) + spinZ(0) + spinZ(1): {(op7 + op8).matrix}')
print(f'(spinX(0) + spinX(1)) * (spinZ(0) + spinZ(1)): {(op7 * op8).matrix}')

print(f'spinX(0) * (spinZ(0) + spinZ(1)): {(spin.x(0) * op8).matrix}')
print(f'(spinZ(0) + spinZ(1)) * spinX(0): {(op8 * spin.x(0)).matrix}')
