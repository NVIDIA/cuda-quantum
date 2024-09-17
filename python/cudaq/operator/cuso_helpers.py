import numpy
from typing import  Mapping, List
from .expressions import ElementaryOperator, ScalarOperator
from .manipulation import OperatorArithmetics
import cusuperop as cuso
from cusuperop._internal.callbacks import CallbackCoefficient

class CuSuperOpHamConversion(OperatorArithmetics[cuso.OperatorTerm | CallbackCoefficient ]):
    def __init__(self, dimensions: Mapping[int, int]):
        self._dimensions = dimensions
    def _scalars_add(self, op1: CallbackCoefficient, op2: CallbackCoefficient) -> CallbackCoefficient:
        if op1.is_callable or op2.is_callable:
            raise ValueError("Cannot add CallbackCoefficients")
        coeff = CallbackCoefficient(None, op1.scalar + op2.scalar)
        return coeff 
    
    
    def _wrap_callback_mult_scalar(self, func, scalar):
        def inplace_func(t, args):
            return func(t, args) * scalar
        return inplace_func
    
    def _wrap_callback_mult(self, func1, func2):
        def inplace_func(t, args):
            return func1(t) * func2(t)
        return inplace_func

    def _scalars_mult(self, op1: CallbackCoefficient, op2: CallbackCoefficient) -> CallbackCoefficient:
        if op1.is_callable and op2.is_callable:
            return CallbackCoefficient(self._wrap_callback_mult(op1.callback, op2.callback))
        elif op1.is_callable:
            if op2.scalar == 1.0:
                return op1
            return CallbackCoefficient(self._wrap_callback_mult_scalar(op1.callback, op2.scalar))
        elif op2.is_callable:
            if op1.scalar == 1.0:
                return op2
            return CallbackCoefficient(self._wrap_callback_mult_scalar(op2.callback, op1.scalar))
        else:
            return CallbackCoefficient(None, op1.scalar * op2.scalar)
        
    def _scalar_mult_op(self, scalar: CallbackCoefficient, op: cuso.OperatorTerm) -> cuso.OperatorTerm:
        if scalar.is_callable:
            new_opterm = cuso.OperatorTerm(dtype=op.dtype)
            for term, coeff in zip(op.terms, op._coefficients):
                if coeff.is_callable:
                    new_coeff = CallbackCoefficient(self._wrap_callback_mult(scalar.callback, coeff.callback))
                    new_opterm.append(term, new_coeff)
                else:
                    if coeff.scalar == 1.0:
                        new_opterm.append(term, scalar.callback)
                    else:
                        new_coeff = CallbackCoefficient(self._wrap_callback_mult_scalar(scalar.callback, coeff.scalar))
                        new_opterm.append(term, new_coeff)
            return new_opterm
        else:
            return op * scalar.scalar

    def tensor(self, op1: cuso.OperatorTerm | CallbackCoefficient, op2: cuso.OperatorTerm | CallbackCoefficient) -> cuso.OperatorTerm | CallbackCoefficient:
        if isinstance(op1, CallbackCoefficient) and isinstance(op2, CallbackCoefficient):
            return self._scalars_mult(op1, op2)
        if isinstance(op1, CallbackCoefficient):
            return self._scalar_mult_op(op1, op2)
        if isinstance(op2, CallbackCoefficient):
            return self._scalar_mult_op(op2, op1)
        new_opterm = cuso.OperatorTerm(dtype=op1.dtype)
        for term2, coeff2 in zip(op2.terms, op2._coefficients):
            for term1, coeff1 in zip(op1.terms, op1._coefficients):
                if coeff1.is_callable or coeff2.is_callable:
                    raise ValueError("Cannot multiply CallbackCoefficients")
                mul_term = term1 + term2
                new_opterm.append(mul_term, coeff1.scalar * coeff2.scalar)
        return new_opterm     


    def mul(self, op1: cuso.OperatorTerm | CallbackCoefficient, op2: cuso.OperatorTerm | CallbackCoefficient) -> cuso.OperatorTerm | CallbackCoefficient:
        if isinstance(op1, CallbackCoefficient) and isinstance(op2, CallbackCoefficient):
            return self._scalars_mult(op1, op2)
        if isinstance(op1, CallbackCoefficient):
            return self._scalar_mult_op(op1, op2)
        if isinstance(op2, CallbackCoefficient):
            return self._scalar_mult_op(op2, op1)
        
        new_opterm = cuso.OperatorTerm(dtype=op1.dtype)
        for term2, coeff2 in zip(op2.terms, op2._coefficients):
            for term1, coeff1 in zip(op1.terms, op1._coefficients):
                if coeff1.is_callable or coeff2.is_callable:
                    raise ValueError("Cannot multiply CallbackCoefficients")
                mul_term = term1 + term2
                new_opterm.append(mul_term, coeff1.scalar * coeff2.scalar)
        return new_opterm

    def add(self, op1: cuso.OperatorTerm | CallbackCoefficient, op2: cuso.OperatorTerm | CallbackCoefficient) -> cuso.OperatorTerm | CallbackCoefficient:
        if isinstance(op1, CallbackCoefficient) and isinstance(op2, CallbackCoefficient):
            return self._scalars_add(op1, op2)
        if isinstance(op1, CallbackCoefficient) or isinstance(op2, CallbackCoefficient):
            raise ValueError("Both add operands must be all scalars or all operators")
        return op1 + op2

    def _wrap_callback(self, func):
        def inplace_func(t, args):
            return func(t)
        return inplace_func
    
    def evaluate(self, op: ElementaryOperator | ScalarOperator) -> cuso.OperatorTerm | CallbackCoefficient: 
        if isinstance(op, ScalarOperator):
            if op._constant_value is None:
                return CallbackCoefficient(self._wrap_callback(op.generator))
            else:
                return CallbackCoefficient(None, op._constant_value)
        else:
            if op._id == "identity":
                return CallbackCoefficient(None, 1.0)
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
