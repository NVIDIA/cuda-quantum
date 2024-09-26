import numpy
from typing import Mapping, List, Sequence
from numbers import Number
from .expressions import ElementaryOperator, ScalarOperator
from .manipulation import OperatorArithmetics
import cusuperop as cuso
from cusuperop._internal.callbacks import CallbackCoefficient
import logging

logger = logging.getLogger(__name__)


class CuSuperOpHamConversion(OperatorArithmetics[cuso.OperatorTerm |
                                                 CallbackCoefficient | Number]):
    """
    Visitor class to convert CUDA-Q operator to a CuSuperOp representation.
    """
    def __init__(self, dimensions: Mapping[int, int]):
        self._dimensions = dimensions

    def _callback_mult_op(self, scalar: CallbackCoefficient,
                          op: cuso.OperatorTerm) -> cuso.OperatorTerm:
        new_opterm = cuso.OperatorTerm(dtype=op.dtype)
        for term, modes, duals, coeff in zip(op.terms, op.modes, op.duals, op._coefficients):
            combined_terms = []
            for sub_op, degrees, duals in zip(term, modes, duals):
                combined_terms.append((sub_op, degrees, duals))
            new_opterm += cuso.tensor_product(*combined_terms, coeff=coeff * scalar)
        return new_opterm

    def tensor(
        self, op1: cuso.OperatorTerm | CallbackCoefficient | Number,
        op2: cuso.OperatorTerm | CallbackCoefficient | Number
    ) -> cuso.OperatorTerm | CallbackCoefficient | Number:
        logger.info(f"Tensor {op1} and {op2}")
        if isinstance(op1, cuso.OperatorTerm):
            logger.info(f" {op1}:")
            for term, coeff in zip(op1.terms, op1._coefficients):
                logger.info(f"  {coeff} * {term}")
        if isinstance(op2, cuso.OperatorTerm):
            logger.info(f" {op2}:")
            for term, coeff in zip(op2.terms, op2._coefficients):
                logger.info(f"  {coeff} * {term}")
        if isinstance(op1, Number) or isinstance(op2, Number):
            return op1 * op2
        if isinstance(op1, CallbackCoefficient) and isinstance(
                op2, CallbackCoefficient):
            return op1 * op2
        if isinstance(op1, CallbackCoefficient):
            return self._callback_mult_op(op1, op2)
        if isinstance(op2, CallbackCoefficient):
            return self._callback_mult_op(op2, op1)
        new_opterm = cuso.OperatorTerm(dtype=op1.dtype)
        for term2, modes2, duals2, coeff2 in zip(op2.terms, op2.modes, op2.duals, op2._coefficients):
            for term1, modes1, duals1, coeff1 in zip(op1.terms, op1.modes, op1.duals, op1._coefficients):
                combined_terms = []
                for sub_op, degrees, duals in zip(term1, modes1, duals1):
                    combined_terms.append((sub_op, degrees, duals))
                for sub_op, degrees, duals in zip(term2, modes2, duals2):
                    combined_terms.append((sub_op, degrees, duals))    
                new_opterm += cuso.tensor_product(*combined_terms, coeff=coeff1 * coeff2)
        return new_opterm

    def mul(
        self, op1: cuso.OperatorTerm | CallbackCoefficient | Number,
        op2: cuso.OperatorTerm | CallbackCoefficient | Number
    ) -> cuso.OperatorTerm | CallbackCoefficient | Number:
        logger.info(f"Multiply {op1} and {op2}")
        if isinstance(op1, cuso.OperatorTerm):
            logger.info(f" {op1}:")
            for term, coeff in zip(op1.terms, op1._coefficients):
                logger.info(f"  {coeff} * {term}")
        if isinstance(op2, cuso.OperatorTerm):
            logger.info(f" {op2}:")
            for term, coeff in zip(op2.terms, op2._coefficients):
                logger.info(f"  {coeff} * {term}")

        if isinstance(op1, Number) or isinstance(op2, Number):
            return op1 * op2
        if isinstance(op1, CallbackCoefficient) and isinstance(
                op2, CallbackCoefficient):
            return op1 * op2
        if isinstance(op1, CallbackCoefficient):
            return self._callback_mult_op(op1, op2)
        if isinstance(op2, CallbackCoefficient):
            return self._callback_mult_op(op2, op1)
        new_opterm = cuso.OperatorTerm(dtype=op1.dtype)
        for term2, modes2, duals2, coeff2 in zip(op2.terms, op2.modes, op2.duals, op2._coefficients):
            for term1, modes1, duals1, coeff1 in zip(op1.terms, op1.modes, op1.duals, op1._coefficients):
                combined_terms = []
                for sub_op, degrees, duals in zip(term1, modes1, duals1):
                    combined_terms.append((sub_op, degrees, duals))
                for sub_op, degrees, duals in zip(term2, modes2, duals2):
                    combined_terms.append((sub_op, degrees, duals))    
                new_opterm += cuso.tensor_product(*combined_terms, coeff=coeff1 * coeff2)
        return new_opterm

    def _scalar_to_op(
            self, scalar: CallbackCoefficient | Number) -> cuso.OperatorTerm:
        op_mat = numpy.identity(self._dimensions[0])
        op_term = cuso.OperatorTerm()
        op_term.append([cuso.TensorOperator(op_mat, 0, (False,))], scalar)
        return op_term

    def add(
        self, op1: cuso.OperatorTerm | CallbackCoefficient | Number,
        op2: cuso.OperatorTerm | CallbackCoefficient | Number
    ) -> cuso.OperatorTerm | CallbackCoefficient | Number:
        logger.info(f"Add {op1} and {op2}")
        if isinstance(op1, cuso.OperatorTerm):
            logger.info(f" {op1}:")
            for term, coeff in zip(op1.terms, op1._coefficients):
                logger.info(f"  {coeff} * {term}")
        if isinstance(op2, cuso.OperatorTerm):
            logger.info(f" {op2}:")
            for term, coeff in zip(op2.terms, op2._coefficients):
                logger.info(f"  {coeff} * {term}")

        if isinstance(op1, Number) and isinstance(op2, Number):
            return op1 + op2
        if isinstance(op1, Number) and op1 == 0.0:
            return op2
        if isinstance(op2, Number) and op2 == 0.0:
            return op1
        if isinstance(op1, CallbackCoefficient) and isinstance(
                op2, CallbackCoefficient):
            return op1 + op2
        return op1 + op2

    def _wrap_callback(self, func):

        def inplace_func(t, args):
            return func(t)

        return inplace_func

    def evaluate(
        self, op: ElementaryOperator | ScalarOperator
    ) -> cuso.OperatorTerm | CallbackCoefficient | Number:
        logger.info(f"Evaluating {op}")
        if isinstance(op, ScalarOperator):
            if op._constant_value is None:
                return CallbackCoefficient(self._wrap_callback(op.generator))
            else:
                return op._constant_value
        else:
            if op._id == "identity":
                return 1.0
            # FIXME: handle callback tensor
            op_mat = op.to_matrix(self._dimensions)
            return cuso.tensor_product((cuso.TensorOperator(op_mat), op.degrees), coeff=1.0)


def computeLindladOp(hilbert_space_dims: List[int], l1: cuso.OperatorTerm,
                     l2: cuso.OperatorTerm):
    """
    Helper function to compute the Lindlad (super-)operator 
    """
    D_terms = []
    
    def conjugate_coeff(coeff: CallbackCoefficient):
        if coeff.is_callable:
            return CallbackCoefficient(lambda t,args : numpy.conjugate(coeff.callback(t,args)))
        return numpy.conjugate(coeff.scalar)

    for term1, modes1, duals1, coeff1 in zip(l1.terms, l1.modes, l1.duals, l1._coefficients):
        for term2, modes2, duals2, coeff2 in zip(l2.terms, l2.modes, l2.duals, l2._coefficients):
            coeff = coeff1 * conjugate_coeff(coeff2)
            d1_terms = []
            for sub_op_1, degrees, duals in zip(term1, modes1, duals1):
                op_mat = sub_op_1._tensor.tensor.numpy()
                d1_terms.append((op_mat, degrees, duals))
            for sub_op_2, degrees, duals in zip(reversed(term2), reversed(modes2), reversed(duals2)):
                op_mat = sub_op_2._tensor.tensor.numpy()
                flipped_duals = tuple((not elem for elem in duals))
                d1_terms.append((numpy.ascontiguousarray(numpy.conj(op_mat).T),
                                 degrees, flipped_duals))
            D1 = cuso.tensor_product(*d1_terms, coeff=coeff)
            D_terms.append(tuple((D1, 1.0)))

            d2_terms = []
            for sub_op_2, degrees, duals in zip(reversed(term2), reversed(modes2), reversed(duals2)):
                op_mat = sub_op_2._tensor.tensor.numpy()
                flipped_duals = tuple((not elem for elem in duals))
                d2_terms.append((numpy.ascontiguousarray(numpy.conj(op_mat).T),
                                 degrees, flipped_duals))
            for sub_op_1, degrees, duals in zip(term1, modes1, duals1):
                op_mat = sub_op_1._tensor.tensor.numpy()
                flipped_duals = tuple((not elem for elem in duals))
                d2_terms.append((op_mat, degrees, flipped_duals))
            D2 = cuso.tensor_product(*d2_terms,
                                     coeff=-0.5 * coeff1 * conjugate_coeff(coeff2))
            D_terms.append(tuple((D2, 1.0)))

            d3_terms = []
            for sub_op_1, degrees, duals in zip(term1, modes1, duals1):
                op_mat = sub_op_1._tensor.tensor.numpy()
                d3_terms.append((op_mat, degrees, duals))
            for sub_op_2, degrees, duals in zip(reversed(term2), reversed(modes2), reversed(duals2)):
                op_mat = sub_op_2._tensor.tensor.numpy()
                d3_terms.append((numpy.ascontiguousarray(numpy.conj(op_mat).T),
                                 degrees, duals))
            D3 = cuso.tensor_product(*d3_terms,
                                     coeff=-0.5 * coeff1 * conjugate_coeff(coeff2))
            D_terms.append(tuple((D3, 1.0)))
    lindblad = cuso.Operator(hilbert_space_dims, *D_terms)
    return lindblad


def constructLiouvillian(hilbert_space_dims: List[int], ham: cuso.OperatorTerm,
                         c_ops: List[cuso.OperatorTerm],
                         is_master_equation: bool):
    """
    Helper to construct the Liouvillian (master or Schrodinger equations) operator
    """
    if not is_master_equation and len(c_ops) > 0:
        raise ValueError(
            "Cannot have collapse operators in non-master equation")
    hamiltonian = cuso.Operator(hilbert_space_dims, (ham, 1.0))
    hamiltonian = hamiltonian * (-1j)
    if is_master_equation:
        liouvillian = hamiltonian - hamiltonian.dual()

        for c_op in c_ops:
            lindbladian = computeLindladOp(hilbert_space_dims, c_op, c_op)
            liouvillian += lindbladian
    else:
        # Schrodinger equation: d/dt psi = -iH psi
        liouvillian = hamiltonian
    return liouvillian
