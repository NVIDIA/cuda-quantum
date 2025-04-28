# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import logging
import numpy
from numbers import Number
from typing import Any, Mapping, List, Union
from ..operators import ElementaryOperator, OperatorArithmetics, ScalarOperator
from .schedule import Schedule
import warnings

cudm = None
CudmStateType = None
try:
    # Suppress deprecation warnings on `cuquantum` import.
    # FIXME: remove this after `cuquantum` no longer warns on import.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from cuquantum import densitymat as cudm
        from cuquantum.densitymat.callbacks import Callback as CallbackCoefficient
        from cuquantum.densitymat.callbacks import CPUCallback
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

    def _scalar_mult(
            self, scalar1: CallbackCoefficient | Number,
            scalar2: CallbackCoefficient | Number
    ) -> CallbackCoefficient | Number:
        if isinstance(scalar1, Number) and isinstance(scalar2, Number):
            return scalar1 * scalar2

        if isinstance(scalar1, CallbackCoefficient) and isinstance(
                scalar2, CallbackCoefficient):
            return CPUCallback(lambda t, args: scalar1.callback(t, args) *
                               scalar2.callback(t, args))

        if isinstance(scalar1, CallbackCoefficient):
            if not isinstance(scalar2, Number):
                raise ValueError(
                    f"Unexpected scalar type {type(scalar2)} in multiplication")
            return CPUCallback(
                lambda t, args: scalar1.callback(t, args) * scalar2)

        if isinstance(scalar2, CallbackCoefficient):
            if not isinstance(scalar1, Number):
                raise ValueError(
                    f"Unexpected scalar type {type(scalar1)} in multiplication")
            return CPUCallback(
                lambda t, args: scalar2.callback(t, args) * scalar1)

        raise ValueError(
            f"Unexpected scalar types: {type(scalar1)} and {type(scalar2)} in multiplication"
        )

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
        if isinstance(op1, Number | CallbackCoefficient) and isinstance(
                op2, Number | CallbackCoefficient):
            return self._scalar_mult(op1, op2)

        # IMPORTANT: `op1 * op2` as written means `op2` to be applied first then `op1`.
        return op2 * op1

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

        if isinstance(op1, Number | CallbackCoefficient) and isinstance(
                op2, Number | CallbackCoefficient):
            return self._scalar_mult(op1, op2)

        # IMPORTANT: `op1 * op2` as written means `op2` to be applied first then `op1`.
        return op2 * op1

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
        if not hasattr(op, "degrees"):
            if not op.is_constant():
                return CPUCallback(
                    self._wrap_callback(lambda **kwargs: op.evaluate(**kwargs),
                                        op.parameters))
            else:
                return op.evaluate()
        else:
            if op == op.__class__(op.degrees[0]):
                return 1.0
            if hasattr(op, "parameters") and len(op.parameters) > 0:
                for param in op.parameters:
                    if param not in self._schedule._parameters:
                        raise ValueError(
                            f"Parameter '{param}' of operator '{op._id}' not found in schedule. Valid schedule parameters are: {self._schedule._parameters}"
                        )
                cudm_callback = CPUCallback(self._wrap_callback_tensor(op))
                c_representative_tensor = cudm_callback(0.0, ())
                return cudm.tensor_product((cudm.DenseOperator(
                    c_representative_tensor, cudm_callback), op.degrees),
                                           coeff=1.0)
            else:
                op_mat = op.to_matrix(self._dimensions)
                return cudm.tensor_product(
                    (cudm.DenseOperator(op_mat), op.degrees), coeff=1.0)


def computeLindladOp(hilbert_space_dims: List[int], l_op: CudmOperatorTerm):
    """
    Helper function to compute the Lindlad (super-)operator 
    """
    term_d1 = l_op * l_op.dag().dual()
    term_d2 = l_op * l_op.dag()
    term_d3 = (term_d2).dual()
    lindblad = cudm.Operator(hilbert_space_dims, (term_d1, 1.0),
                             (term_d2, -0.5), (term_d3, -0.5))
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
            lindbladian = computeLindladOp(hilbert_space_dims, c_op)
            liouvillian += lindbladian
    else:
        # Schrodinger equation: `d/dt psi = -iH psi`
        liouvillian = hamiltonian
    return liouvillian
