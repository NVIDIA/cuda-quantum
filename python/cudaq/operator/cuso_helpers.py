import numpy
from typing import  Mapping, List, Sequence
from numbers import Number
from .expressions import ElementaryOperator, ScalarOperator
from .manipulation import OperatorArithmetics
import cusuperop as cuso
from cusuperop._internal.callbacks import CallbackCoefficient
import logging

logger = logging.getLogger(__name__)

class CuSuperOpHamConversion(OperatorArithmetics[cuso.OperatorTerm | CallbackCoefficient | Number ]):
    def __init__(self, dimensions: Mapping[int, int]):
        self._dimensions = dimensions
        
    def _callback_mult_op(self, scalar: CallbackCoefficient, op: cuso.OperatorTerm) -> cuso.OperatorTerm:
        new_opterm = cuso.OperatorTerm(dtype=op.dtype)
        for term, coeff in zip(op.terms, op._coefficients):
            if coeff.is_callable and scalar.is_callable:
                new_opterm.append(term, coeff * scalar)
            elif not coeff.is_callable and not scalar.is_callable:
                new_opterm.append(term, coeff.scalar * scalar.scalar)
            elif coeff.is_callable:
                new_opterm.append(term, coeff * scalar.scalar)
            elif scalar.is_callable:
                new_opterm.append(term, coeff.scalar * scalar)
            else:
                raise ValueError(f"Unsupported operand types {coeff} and {scalar}")
        return new_opterm

    def tensor(self, op1: cuso.OperatorTerm | CallbackCoefficient | Number, op2: cuso.OperatorTerm | CallbackCoefficient | Number) -> cuso.OperatorTerm | CallbackCoefficient | Number:
        logger.debug(f"Tensor {op1} and {op2}")
        if isinstance(op1, Number) or isinstance(op2, Number):
            return op1 * op2
        if isinstance(op1, CallbackCoefficient) and isinstance(op2, CallbackCoefficient):
            return op1 * op2
        if isinstance(op1, CallbackCoefficient):
            return self._callback_mult_op(op1, op2)
        if isinstance(op2, CallbackCoefficient):
            return self._callback_mult_op(op2, op1)
        new_opterm = cuso.OperatorTerm(dtype=op1.dtype)
        for term2, coeff2 in zip(op2.terms, op2._coefficients):
            for term1, coeff1 in zip(op1.terms, op1._coefficients):
                # FIXME: workaround a bug in cusuperop (CallbackCoefficient.__mul__)
                mul_term = term1 + term2
                if coeff1.is_callable and coeff2.is_callable:
                    new_opterm.append(mul_term, coeff1 * coeff2)
                elif not coeff1.is_callable and not coeff2.is_callable:
                    new_opterm.append(mul_term, coeff1.scalar * coeff2.scalar)
                elif coeff1.is_callable:
                    new_opterm.append(mul_term, coeff1 * coeff2.scalar)
                elif coeff2.is_callable:
                    new_opterm.append(mul_term, coeff1.scalar * coeff2)
                else:
                    raise ValueError(f"Unsupported operand types {coeff1} and {coeff2}")
        return new_opterm     


    def mul(self, op1: cuso.OperatorTerm | CallbackCoefficient | Number, op2: cuso.OperatorTerm | CallbackCoefficient | Number) -> cuso.OperatorTerm | CallbackCoefficient | Number:
        logger.debug(f"Multiply {op1} and {op2}")
        if isinstance(op1, Number) or isinstance(op2, Number):
            return op1 * op2
        if isinstance(op1, CallbackCoefficient) and isinstance(op2, CallbackCoefficient):
            return op1 * op2
        if isinstance(op1, CallbackCoefficient):
            return self._callback_mult_op(op1, op2)
        if isinstance(op2, CallbackCoefficient):
            return self._callback_mult_op(op2, op1)
        new_opterm = cuso.OperatorTerm(dtype=op1.dtype)
        for term2, coeff2 in zip(op2.terms, op2._coefficients):
            for term1, coeff1 in zip(op1.terms, op1._coefficients):
                # FIXME: workaround a bug in cusuperop (CallbackCoefficient.__mul__)
                mul_term = term1 + term2
                if coeff1.is_callable and coeff2.is_callable:
                    new_opterm.append(mul_term, coeff1 * coeff2)
                elif not coeff1.is_callable and not coeff2.is_callable:
                    new_opterm.append(mul_term, coeff1.scalar * coeff2.scalar)
                elif coeff1.is_callable:
                    new_opterm.append(mul_term, coeff1 * coeff2.scalar)
                elif coeff2.is_callable:
                    new_opterm.append(mul_term, coeff1.scalar * coeff2)
                else:
                    raise ValueError(f"Unsupported operand types {coeff1} and {coeff2}")
        return new_opterm

    def _scalar_to_op(self, scalar: CallbackCoefficient | Number) -> cuso.OperatorTerm:
        op_mat = numpy.identity(self._dimensions[0])
        op_term = cuso.OperatorTerm()
        op_term.append([cuso.TensorOperator(op_mat, 0, (False,))], scalar)
        return op_term
    
    def add(self, op1: cuso.OperatorTerm | CallbackCoefficient | Number, op2: cuso.OperatorTerm | CallbackCoefficient | Number) -> cuso.OperatorTerm | CallbackCoefficient | Number:
        logger.debug(f"Add {op1} and {op2}")
        if isinstance(op1, Number) and isinstance(op2, Number):
            return op1 + op2
        if isinstance(op1, Number) and op1 == 0.0:
            return op2
        if isinstance(op2, Number) and op2 == 0.0:
            return op1
        if isinstance(op1, CallbackCoefficient) and isinstance(op2, CallbackCoefficient):
            return op1 + op2
        return op1 + op2

    def _wrap_callback(self, func):
        def inplace_func(t, args):
            return func(t)
        return inplace_func
    
    def evaluate(self, op: ElementaryOperator | ScalarOperator) -> cuso.OperatorTerm | CallbackCoefficient | Number: 
        logger.debug(f"Evaluating {op}")
        if isinstance(op, ScalarOperator):
            if op._constant_value is None:
                return CallbackCoefficient(self._wrap_callback(op.generator))
            else:
                return op._constant_value
        else:
            if op._id == "identity":
                return 1.0
            op_mat = op.to_matrix(self._dimensions)
            op_term = cuso.OperatorTerm()
            op_term.append([cuso.TensorOperator(op_mat, op.degrees, ((False,) * len(op.degrees)))], 1.0)
            return op_term


def computeLindladOp(hilbert_space_dims: List[int], l1: cuso.OperatorTerm, l2: cuso.OperatorTerm):
    D_terms = []
    for term1, coeff1 in zip(l1.terms, l1._coefficients):
        for term2, coeff2 in zip(l2.terms, l2._coefficients):
            if coeff1.is_callable or coeff2.is_callable:
                raise ValueError("Cannot multiply CallbackCoefficients")
            coeff = coeff1.scalar * numpy.conjugate(coeff2.scalar)
            d1_terms = []
            for sub_op_1 in term1:
                op_mat = sub_op_1._tensor.tensor.numpy()
                degrees = sub_op_1.modes
                d1_terms.append((op_mat, degrees, (False,)))
            for sub_op_2 in reversed(term2):
                op_mat = sub_op_2._tensor.tensor.numpy()
                degrees = sub_op_2.modes
                d1_terms.append((numpy.ascontiguousarray(numpy.conj(op_mat).T), degrees, (True,)))
            D1 = cuso.tensor_product(*d1_terms, coeff = coeff)
            D_terms.append(tuple((D1, 1.0)))
            
            d2_terms = []
            for sub_op_2 in reversed(term2):
                op_mat = sub_op_2._tensor.tensor.numpy()
                degrees = sub_op_2.modes
                d2_terms.append((numpy.ascontiguousarray(numpy.conj(op_mat).T), degrees, (True,)))
            for sub_op_1 in term1:
                op_mat = sub_op_1._tensor.tensor.numpy()
                degrees = sub_op_1.modes
                d2_terms.append((op_mat, degrees, (True,)))
            D2 = cuso.tensor_product(*d2_terms, coeff = -0.5 * coeff1.scalar * numpy.conjugate(coeff2.scalar))
            D_terms.append(tuple((D2, 1.0)))
            
            d3_terms = []
            for sub_op_1 in term1:
                op_mat = sub_op_1._tensor.tensor.numpy()
                degrees = sub_op_1.modes
                d3_terms.append((op_mat, degrees, (False,)))
            for sub_op_2 in reversed(term2):
                op_mat = sub_op_2._tensor.tensor.numpy()
                degrees = sub_op_2.modes
                d3_terms.append((numpy.ascontiguousarray(numpy.conj(op_mat).T), degrees, (False,)))
            D3 = cuso.tensor_product(*d3_terms, coeff = -0.5 * coeff1.scalar * numpy.conjugate(coeff2.scalar))
            D_terms.append(tuple((D3, 1.0)))
    lindblad = cuso.Operator(hilbert_space_dims, *D_terms)
    return lindblad


def constructLiouvillian(hilbert_space_dims: List[int], ham: cuso.OperatorTerm, c_ops: List[cuso.OperatorTerm]):
    hamiltonian = cuso.Operator(hilbert_space_dims, (ham, 1.0))
    hamiltonian = hamiltonian * (-1j)
    liouvillian = hamiltonian - hamiltonian.dual() 
    
    for c_op in c_ops:
        lindbladian = computeLindladOp(hilbert_space_dims, c_op, c_op)
        liouvillian += lindbladian

    return liouvillian
