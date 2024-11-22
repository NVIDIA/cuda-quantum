# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy
from typing import Any, Mapping, List, Sequence, Union
from numbers import Number
from .expressions import ElementaryOperator, ScalarOperator
from .manipulation import OperatorArithmetics
import logging
from .schedule import Schedule

cudm = None
CudmStateType = None
try:
    from cuquantum import densitymat as cudm
    from cuquantum.densitymat._internal.callbacks import CallbackCoefficient
    CudmStateType = Union[cudm.DensePureState, cudm.DenseMixedState]
    CudmOperator = cudm.Operator
    CudmOperatorTerm = cudm.OperatorTerm
    CudmWorkStream = cudm.WorkStream
except ImportError:
    cudm = None
    CudmOperator = Any
    CudmOperatorTerm = Any
    CudmWorkStream = Any
    CallbackCoefficient = Any

logger = logging.getLogger(__name__)


class CuDensityMatOpConversion(
        OperatorArithmetics[CudmOperatorTerm | CallbackCoefficient | Number]):
    """
    Visitor class to convert CUDA-Q operator to a `cuquantum` representation.
    """

    def __init__(self,
                 dimensions: Mapping[int, int],
                 schedule: Schedule = None):
        self._dimensions = dimensions
        self._schedule = schedule

    def _callback_mult_op(self, scalar: CallbackCoefficient,
                          op: CudmOperatorTerm) -> CudmOperatorTerm:
        new_opterm = CudmOperatorTerm(dtype=op.dtype)
        for term, modes, duals, coeff in zip(op.terms, op.modes, op.duals,
                                             op._coefficients):
            combined_terms = []
            for sub_op, degrees, duals in zip(term, modes, duals):
                combined_terms.append((sub_op, degrees, duals))
            new_opterm += cudm.tensor_product(*combined_terms,
                                              coeff=coeff * scalar)
        return new_opterm

    def tensor(
        self, op1: CudmOperatorTerm | CallbackCoefficient | Number,
        op2: CudmOperatorTerm | CallbackCoefficient | Number
    ) -> CudmOperatorTerm | CallbackCoefficient | Number:
        logger.info(f"Tensor {op1} and {op2}")
        if isinstance(op1, CudmOperatorTerm):
            logger.info(f" {op1}:")
            for term, coeff in zip(op1.terms, op1._coefficients):
                logger.info(f"  {coeff} * {term}")
        if isinstance(op2, CudmOperatorTerm):
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
        new_opterm = CudmOperatorTerm(dtype=op1.dtype)
        for term2, modes2, duals2, coeff2 in zip(op2.terms, op2.modes,
                                                 op2.duals, op2._coefficients):
            for term1, modes1, duals1, coeff1 in zip(op1.terms, op1.modes,
                                                     op1.duals,
                                                     op1._coefficients):
                combined_terms = []
                for sub_op, degrees, duals in zip(term1, modes1, duals1):
                    combined_terms.append((sub_op, degrees, duals))
                for sub_op, degrees, duals in zip(term2, modes2, duals2):
                    combined_terms.append((sub_op, degrees, duals))
                new_opterm += cudm.tensor_product(*combined_terms,
                                                  coeff=coeff1 * coeff2)
        return new_opterm

    def mul(
        self, op1: CudmOperatorTerm | CallbackCoefficient | Number,
        op2: CudmOperatorTerm | CallbackCoefficient | Number
    ) -> CudmOperatorTerm | CallbackCoefficient | Number:
        logger.info(f"Multiply {op1} and {op2}")
        if isinstance(op1, CudmOperatorTerm):
            logger.info(f" {op1}:")
            for term, coeff in zip(op1.terms, op1._coefficients):
                logger.info(f"  {coeff} * {term}")
        if isinstance(op2, CudmOperatorTerm):
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
        new_opterm = CudmOperatorTerm(dtype=op1.dtype)
        for term2, modes2, duals2, coeff2 in zip(op2.terms, op2.modes,
                                                 op2.duals, op2._coefficients):
            for term1, modes1, duals1, coeff1 in zip(op1.terms, op1.modes,
                                                     op1.duals,
                                                     op1._coefficients):
                combined_terms = []
                for sub_op, degrees, duals in zip(term2, modes2, duals2):
                    combined_terms.append((sub_op, degrees, duals))

                for sub_op, degrees, duals in zip(term1, modes1, duals1):
                    combined_terms.append((sub_op, degrees, duals))

                new_opterm += cudm.tensor_product(*combined_terms,
                                                  coeff=coeff1 * coeff2)
        return new_opterm

    def _scalar_to_op(self,
                      scalar: CallbackCoefficient | Number) -> CudmOperatorTerm:
        op_mat = numpy.identity(self._dimensions[0], dtype=numpy.complex128)
        op_term = cudm.tensor_product((cudm.DenseOperator(op_mat), (0,)),
                                      coeff=scalar)
        return op_term

    def add(
        self, op1: CudmOperatorTerm | CallbackCoefficient | Number,
        op2: CudmOperatorTerm | CallbackCoefficient | Number
    ) -> CudmOperatorTerm | CallbackCoefficient | Number:
        logger.info(f"Add {op1} and {op2}")
        if isinstance(op1, CudmOperatorTerm):
            logger.info(f" {op1}:")
            for term, coeff in zip(op1.terms, op1._coefficients):
                logger.info(f"  {coeff} * {term}")
        if isinstance(op2, CudmOperatorTerm):
            logger.info(f" {op2}:")
            for term, coeff in zip(op2.terms, op2._coefficients):
                logger.info(f"  {coeff} * {term}")

        if isinstance(op1, Number) and isinstance(op2, Number):
            return op1 + op2
        if isinstance(op1, Number) and op1 == 0.0:
            return op2
        if isinstance(op2, Number) and op2 == 0.0:
            return op1
        if isinstance(op1, Number) and isinstance(op2, CudmOperatorTerm):
            return op2 + self._scalar_to_op(op1)
        if isinstance(op2, Number) and isinstance(op1, CudmOperatorTerm):
            return op1 + self._scalar_to_op(op2)
        if isinstance(op1, CallbackCoefficient) and isinstance(
                op2, CallbackCoefficient):
            return op1 + op2
        return op1 + op2

    def _wrap_callback(self, func, params):
        for param in params:
            if param not in self._schedule._parameters:
                raise ValueError(
                    f"Parameter '{param}' not found in schedule. Valid schedule parameters are: {self._schedule._parameters}"
                )

        def inplace_func(t, args):
            call_params = dict(
                ((parameter, self._schedule._get_value(parameter, t))
                 for parameter in self._schedule._parameters))
            try:
                return func(**call_params)
            except Exception as e:
                print(f"Error in callback function: {e}")
                raise RuntimeError("Failed to execute callback function")

        return inplace_func

    def _wrap_callback_tensor(self, op):

        def c_callback(t, args):
            try:
                call_params = dict(
                    ((parameter, self._schedule._get_value(parameter, t))
                     for parameter in self._schedule._parameters))
                op_mat = op.to_matrix(self._dimensions, **call_params)
                return op_mat
            except Exception as e:
                print(f"Error in callback function: {e}")
                raise RuntimeError("Failed to execute callback function")

        return c_callback

    def evaluate(
        self, op: ElementaryOperator | ScalarOperator
    ) -> CudmOperatorTerm | CallbackCoefficient | Number:
        logger.info(f"Evaluating {op}")
        if isinstance(op, ScalarOperator):
            if op._constant_value is None:
                return CallbackCoefficient(
                    self._wrap_callback(op.generator, op.parameters))
            else:
                return op._constant_value
        else:
            if op._id == "identity":
                return 1.0
            if len(op.parameters) > 0:
                for param in op.parameters:
                    if param not in self._schedule._parameters:
                        raise ValueError(
                            f"Parameter '{param}' of operator '{op._id}' not found in schedule. Valid schedule parameters are: {self._schedule._parameters}"
                        )
                cudm_callback = self._wrap_callback_tensor(op)
                c_representative_tensor = cudm_callback(0.0, ())
                return cudm.tensor_product((cudm.DenseOperator(
                    c_representative_tensor, cudm_callback), op.degrees),
                                           coeff=1.0)
            else:
                op_mat = op.to_matrix(self._dimensions)
                return cudm.tensor_product(
                    (cudm.DenseOperator(op_mat), op.degrees), coeff=1.0)


def computeLindladOp(hilbert_space_dims: List[int], l1: CudmOperatorTerm,
                     l2: CudmOperatorTerm):
    """
    Helper function to compute the Lindlad (super-)operator 
    """
    D_terms = []

    def conjugate_coeff(coeff: CallbackCoefficient):
        if coeff.is_callable:
            return CallbackCoefficient(
                lambda t, args: numpy.conjugate(coeff.callback(t, args)))
        return numpy.conjugate(coeff.scalar)

    for term1, modes1, duals1, coeff1 in zip(l1.terms, l1.modes, l1.duals,
                                             l1._coefficients):
        for term2, modes2, duals2, coeff2 in zip(l2.terms, l2.modes, l2.duals,
                                                 l2._coefficients):
            coeff = coeff1 * conjugate_coeff(coeff2)
            d1_terms = []
            for sub_op_1, degrees, duals in zip(term1, modes1, duals1):
                op_mat = sub_op_1.to_array()
                d1_terms.append((op_mat, degrees, duals))
            for sub_op_2, degrees, duals in zip(reversed(term2),
                                                reversed(modes2),
                                                reversed(duals2)):
                op_mat = sub_op_2.to_array()
                flipped_duals = tuple((not elem for elem in duals))
                d1_terms.append(
                    (numpy.ascontiguousarray(numpy.conj(op_mat).T).copy(),
                     degrees, flipped_duals))
            D1 = cudm.tensor_product(*d1_terms, coeff=coeff)
            D_terms.append(tuple((D1, 1.0)))

            d2_terms = []
            for sub_op_2, degrees, duals in zip(reversed(term2),
                                                reversed(modes2),
                                                reversed(duals2)):
                op_mat = sub_op_2.to_array()
                flipped_duals = tuple((not elem for elem in duals))
                d2_terms.append(
                    (numpy.ascontiguousarray(numpy.conj(op_mat).T).copy(),
                     degrees, flipped_duals))
            for sub_op_1, degrees, duals in zip(term1, modes1, duals1):
                op_mat = sub_op_1.to_array()
                flipped_duals = tuple((not elem for elem in duals))
                d2_terms.append((op_mat, degrees, flipped_duals))
            D2 = cudm.tensor_product(*d2_terms,
                                     coeff=-0.5 * coeff1 *
                                     conjugate_coeff(coeff2))
            D_terms.append(tuple((D2, 1.0)))

            d3_terms = []
            for sub_op_1, degrees, duals in zip(term1, modes1, duals1):
                op_mat = sub_op_1.to_array()
                d3_terms.append((op_mat, degrees, duals))
            for sub_op_2, degrees, duals in zip(reversed(term2),
                                                reversed(modes2),
                                                reversed(duals2)):
                op_mat = sub_op_2.to_array()
                d3_terms.append(
                    (numpy.ascontiguousarray(numpy.conj(op_mat).T).copy(),
                     degrees, duals))
            D3 = cudm.tensor_product(*d3_terms,
                                     coeff=-0.5 * coeff1 *
                                     conjugate_coeff(coeff2))
            D_terms.append(tuple((D3, 1.0)))
    lindblad = cudm.Operator(hilbert_space_dims, *D_terms)
    return lindblad


def constructLiouvillian(hilbert_space_dims: List[int], ham: CudmOperatorTerm,
                         c_ops: List[CudmOperatorTerm],
                         is_master_equation: bool):
    """
    Helper to construct the Liouvillian (master or Schrodinger equations) operator
    """
    if not is_master_equation and len(c_ops) > 0:
        raise ValueError(
            "Cannot have collapse operators in non-master equation")
    hamiltonian = cudm.Operator(hilbert_space_dims, (ham, 1.0))
    hamiltonian = hamiltonian * (-1j)
    if is_master_equation:
        liouvillian = hamiltonian - hamiltonian.dual()

        for c_op in c_ops:
            lindbladian = computeLindladOp(hilbert_space_dims, c_op, c_op)
            liouvillian += lindbladian
    else:
        # Schrodinger equation: `d/dt psi = -iH psi`
        liouvillian = hamiltonian
    return liouvillian
