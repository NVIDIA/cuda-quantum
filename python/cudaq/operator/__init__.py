import scipy
from .definitions import operators, pauli
from .expressions import OperatorSum, ProductOperator, ElementaryOperator, ScalarOperator
from .schedule import Schedule

Operator = OperatorSum | ProductOperator | ElementaryOperator | ScalarOperator

def create_time_evolution(op: Operator, dimensions, **kwargs):
    op_matrix = op.to_matrix(dimensions, **kwargs)
    return scipy.linalg.expm(-1j * op_matrix)

