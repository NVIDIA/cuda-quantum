import scipy

from .definitions import operators, pauli
from .evolution import EvolveResult
from .expressions import Operator, OperatorSum, ProductOperator, ElementaryOperator, ScalarOperator
from .schedule import Schedule


def create_time_evolution(op: Operator, dimensions, **kwargs):
    op_matrix = op.to_matrix(dimensions, **kwargs)
    return scipy.linalg.expm(-1j * op_matrix)

