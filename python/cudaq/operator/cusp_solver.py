from __future__ import annotations
import numpy, scipy, sys, uuid
from typing import Callable, Mapping, Optional, Sequence, List

from .expressions import Operator, OperatorSum, ProductOperator, ElementaryOperator, ScalarOperator
from .helpers import _OperatorHelpers
from .schedule import Schedule
from ..kernel.register_op import register_operation
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
from ..kernel.kernel_decorator import PyKernelDecorator
from ..runtime.observe import observe
from .manipulation import OperatorArithmetics

import cupy as cp
import numpy as np

import cusuperop as cuso

from enum import Enum

class CuSuperOpHamConversion(OperatorArithmetics[cuso.OperatorTerm | cuso.CallbackCoefficient ]):
    def __init__(self, dimensions: Mapping[int, int]):
        self._dimensions = dimensions
    def _scalars_add(self, op1: cuso.CallbackCoefficient, op2: cuso.CallbackCoefficient) -> cuso.CallbackCoefficient:
        if op1.is_callable or op2.is_callable:
            raise ValueError("Cannot add CallbackCoefficients")
        coeff = cuso.CallbackCoefficient(None, op1.scalar + op2.scalar)
        return coeff 
    
    
    def _wrap_callback_mult_scalar(self, func, scalar):
        def inplace_func(t, args):
            return func(t, args) * scalar
        return inplace_func
    
    def _wrap_callback_mult(self, func1, func2):
        def inplace_func(t, args):
            return func1(t) * func2(t)
        return inplace_func

    def _scalars_mult(self, op1: cuso.CallbackCoefficient, op2: cuso.CallbackCoefficient) -> cuso.CallbackCoefficient:
        if op1.is_callable and op2.is_callable:
            return cuso.CallbackCoefficient(self._wrap_callback_mult(op1.callback, op2.callback))
        elif op1.is_callable:
            if op2.scalar == 1.0:
                return op1
            return cuso.CallbackCoefficient(self._wrap_callback_mult_scalar(op1.callback, op2.scalar))
        elif op2.is_callable:
            if op1.scalar == 1.0:
                return op2
            return cuso.CallbackCoefficient(self._wrap_callback_mult_scalar(op2.callback, op1.scalar))
        else:
            return cuso.CallbackCoefficient(None, op1.scalar * op2.scalar)
        
    def _scalar_mult_op(self, scalar: cuso.CallbackCoefficient, op: cuso.OperatorTerm) -> cuso.OperatorTerm:
        if scalar.is_callable:
            new_opterm = cuso.OperatorTerm(dtype=op.dtype)
            for term, coeff in zip(op.terms, op.coefficients):
                if coeff.is_callable:
                    new_coeff = cuso.CallbackCoefficient(self._wrap_callback_mult(scalar.callback, coeff.callback))
                    new_opterm.append(term, new_coeff)
                else:
                    if coeff.scalar == 1.0:
                        new_opterm.append(term, scalar.callback)
                    else:
                        new_coeff = cuso.CallbackCoefficient(self._wrap_callback_mult_scalar(scalar.callback, coeff.scalar))
                        new_opterm.append(term, new_coeff)
            return new_opterm
        else:
            return op * scalar.scalar

    def tensor(self, op1: cuso.OperatorTerm | cuso.CallbackCoefficient, op2: cuso.OperatorTerm | cuso.CallbackCoefficient) -> cuso.OperatorTerm | cuso.CallbackCoefficient:
        if isinstance(op1, cuso.CallbackCoefficient) and isinstance(op2, cuso.CallbackCoefficient):
            return self._scalars_mult(op1, op2)
        if isinstance(op1, cuso.CallbackCoefficient):
            return self._scalar_mult_op(op1, op2)
        if isinstance(op2, cuso.CallbackCoefficient):
            return self._scalar_mult_op(op2, op1)
        new_opterm = cuso.OperatorTerm(dtype=op1.dtype)
        for term2, coeff2 in zip(op2.terms, op2.coefficients):
            for term1, coeff1 in zip(op1.terms, op1.coefficients):
                if coeff1.is_callable or coeff2.is_callable:
                    raise ValueError("Cannot multiply CallbackCoefficients")
                mul_term = term1 + term2
                new_opterm.append(mul_term, coeff1.scalar * coeff2.scalar)
        return new_opterm     


    def mul(self, op1: cuso.OperatorTerm | cuso.CallbackCoefficient, op2: cuso.OperatorTerm | cuso.CallbackCoefficient) -> cuso.OperatorTerm | cuso.CallbackCoefficient:
        if isinstance(op1, cuso.CallbackCoefficient) and isinstance(op2, cuso.CallbackCoefficient):
            return self._scalars_mult(op1, op2)
        if isinstance(op1, cuso.CallbackCoefficient):
            return self._scalar_mult_op(op1, op2)
        if isinstance(op2, cuso.CallbackCoefficient):
            return self._scalar_mult_op(op2, op1)
        
        new_opterm = cuso.OperatorTerm(dtype=op1.dtype)
        for term2, coeff2 in zip(op2.terms, op2.coefficients):
            for term1, coeff1 in zip(op1.terms, op1.coefficients):
                if coeff1.is_callable or coeff2.is_callable:
                    raise ValueError("Cannot multiply CallbackCoefficients")
                mul_term = term1 + term2
                new_opterm.append(mul_term, coeff1.scalar * coeff2.scalar)
        return new_opterm

    def add(self, op1: cuso.OperatorTerm | cuso.CallbackCoefficient, op2: cuso.OperatorTerm | cuso.CallbackCoefficient) -> cuso.OperatorTerm | cuso.CallbackCoefficient:
        if isinstance(op1, cuso.CallbackCoefficient) and isinstance(op2, cuso.CallbackCoefficient):
            return self._scalars_add(op1, op2)
        if isinstance(op1, cuso.CallbackCoefficient) or isinstance(op2, cuso.CallbackCoefficient):
            raise ValueError("Both add operands must be all scalars or all operators")
        return op1 + op2

    def _wrap_callback(self, func):
        def inplace_func(t, args):
            return func(t)
        return inplace_func
    
    def evaluate(self, op: ElementaryOperator | ScalarOperator) -> cuso.OperatorTerm | cuso.CallbackCoefficient: 
        if isinstance(op, ScalarOperator):
            if op._constant_value is None:
                return cuso.CallbackCoefficient(self._wrap_callback(op.generator))
            else:
                return cuso.CallbackCoefficient(None, op._constant_value)
        else:
            if op._id == "identity":
                return cuso.CallbackCoefficient(None, 1.0)
            op_mat = op.to_matrix(self._dimensions)
            op_term = cuso.OperatorTerm()
            op_term.append([cuso.GeneralOperator(cuso.CallbackTensor(op_mat), op.degrees, ((False,) * len(op.degrees)))], 1.0)
            return op_term


def computeLindladOp(hilbert_space_dims: List[int], l1: cuso.OperatorTerm, l2: cuso.OperatorTerm):
    D_terms = []
    for term1, coeff1 in zip(l1.terms, l1.coefficients):
        for term2, coeff2 in zip(l2.terms, l2.coefficients):
            if coeff1.is_callable or coeff2.is_callable:
                raise ValueError("Cannot multiply CallbackCoefficients")
            coeff = coeff1.scalar * numpy.conjugate(coeff2.scalar)
            d1_terms = []
            for sub_op_1 in term1:
                op_mat = sub_op_1.tensor.numpy()
                degrees = sub_op_1.modes
                d1_terms.append((op_mat, degrees, (False,)))
            for sub_op_2 in reversed(term2):
                op_mat = sub_op_2.tensor.numpy()
                degrees = sub_op_2.modes
                d1_terms.append((numpy.conj(op_mat).T, degrees, (True,)))
            D1 = cuso.tensor_product(*d1_terms, coeff = coeff)
            D_terms.append(tuple((D1, 1.0)))
            
            d2_terms = []
            for sub_op_2 in reversed(term2):
                op_mat = sub_op_2.tensor.numpy()
                degrees = sub_op_2.modes
                d2_terms.append((numpy.conj(op_mat).T, degrees, (True,)))
            for sub_op_1 in term1:
                op_mat = sub_op_1.tensor.numpy()
                degrees = sub_op_1.modes
                d2_terms.append((op_mat, degrees, (True,)))
            D2 = cuso.tensor_product(*d2_terms, coeff = -0.5 * coeff1.scalar * numpy.conjugate(coeff2.scalar))
            D_terms.append(tuple((D2, 1.0)))
            
            d3_terms = []
            for sub_op_1 in term1:
                op_mat = sub_op_1.tensor.numpy()
                degrees = sub_op_1.modes
                d3_terms.append((op_mat, degrees, (False,)))
            for sub_op_2 in reversed(term2):
                op_mat = sub_op_2.tensor.numpy()
                degrees = sub_op_2.modes
                d3_terms.append((numpy.conj(op_mat).T, degrees, (False,)))
            D3 = cuso.tensor_product(*d3_terms, coeff = -0.5 * coeff1.scalar * numpy.conjugate(coeff2.scalar))
            D_terms.append(tuple((D3, 1.0)))
    lindblad = cuso.Operator(hilbert_space_dims, *D_terms)
    return lindblad


def constructLiouvillian(hilbert_space_dims: List[int], ham: cuso.OperatorTerm, c_ops: List[cuso.OperatorTerm]):
    hamiltonian = cuso.Operator(hilbert_space_dims, (ham, 1.0))
    hamiltonian = hamiltonian * (-1j)
    liouvillian = hamiltonian - hamiltonian.dual() 
    
    for c1 in c_ops:
        for c2 in c_ops:
            lindbladian = computeLindladOp(hilbert_space_dims, c1, c2)
            liouvillian += lindbladian

    return liouvillian

class OpConversionMode(Enum):
    Hamiltonian = 1
    Lindbladian = 2

# Helper to convert to cuSuperOp format
def to_cusp_operator(operator: Operator, dimensions: Mapping[int, int], mode: OpConversionMode = OpConversionMode.Hamiltonian) -> cuso.Operator:
    hilbert_space_dims = tuple(dimensions[d] for d in range(len(dimensions)))
    ham_term = operator._evaluate(CuSuperOpHamConversion(dimensions))
    hamiltonian = cuso.Operator(hilbert_space_dims, (ham_term, 1.0))
    return hamiltonian
    # print(f"Op: {operator}, type: {type(operator)}")
    if type(operator) == ElementaryOperator:
        ham_term = cuso.OperatorTerm()
        if len(operator.parameters) > 0:
            raise NotImplementedError("TODO: implement ElementaryOperator with parameters")
        op_mat = operator.to_matrix(dimensions)
        if mode == OpConversionMode.Hamiltonian:
            ham_term.append([cuso.GeneralOperator(cuso.CallbackTensor(op_mat), operator.degrees, ((False,) * len(operator.degrees)))], 1.0)
            hamiltonian = cuso.Operator(hilbert_space_dims, (ham_term, 1.0))
            return hamiltonian
        else:
            L = op_mat
            L_dag = np.conj(L).T
            D1 = cuso.tensor_product(   # an operator term composed of a single elementary tensor operator
            (
                L,        # elementary tensor operator
                (0,),     # quantum degrees of freedom it acts on
                (False,)  # operator action duality (side: left/right) for each quantum degree of freedom
            ),
            (
                L_dag,        # elementary tensor operator
                (0,),     # quantum degrees of freedom it acts on
                (True,)  # operator action duality (side: left/right) for each quantum degree of freedom
            ),
            coeff = 1.0   # constant (static) coefficient
            )


            D2 = cuso.tensor_product(   # an operator term composed of a single elementary tensor operator
                (
                    L_dag,        # elementary tensor operator
                    (0,),     # quantum degrees of freedom it acts on
                    (True,)  # operator action duality (side: left/right) for each quantum degree of freedom
                ),
                (
                    L,        # elementary tensor operator
                    (0,),     # quantum degrees of freedom it acts on
                    (True,)  # operator action duality (side: left/right) for each quantum degree of freedom
                ),
                coeff = -0.5  # constant (static) coefficient
            )

            D3 = cuso.tensor_product(   # an operator term composed of a single elementary tensor operator
                (
                    L_dag,        # elementary tensor operator
                    (0,),     # quantum degrees of freedom it acts on
                    (False,)  # operator action duality (side: left/right) for each quantum degree of freedom
                ),
                (
                    L,        # elementary tensor operator
                    (0,),     # quantum degrees of freedom it acts on
                    (False,)  # operator action duality (side: left/right) for each quantum degree of freedom
                ),
                coeff = -0.5  # constant (static) coefficient
            )

            lindblad = cuso.Operator(hilbert_space_dims, (D1, 1.0), (D2, 1.0), (D3, 1.0))
            return lindblad
    elif type(operator) == ProductOperator:
        coeff = None
        cusp_terms = []
        for sub_term in operator._operators:
            # print(f"Sub-term: {sub_term}, type: {type(sub_term)}")
            if type(sub_term) == ScalarOperator:
                if sub_term._constant_value is None:
                    # print(f"Generator = {sub_term.generator}")
                    # print(f"Parameters = {sub_term.parameters}")
                    coeff = sub_term.generator
                else:
                    coeff = sub_term._constant_value
            elif type(sub_term) == ElementaryOperator:
                op_mat = cuso.optimize_strides(sub_term.to_matrix(dimensions))
                # print(f"Op mat: {op_mat.shape}")
                # print(f"Degree = {sub_term.degrees}")
                cusp_terms.append(tuple((op_mat, sub_term.degrees)))
            else:
                raise NotImplementedError(f"Unsupported operator type: {type(sub_term)}")
        
        if mode == OpConversionMode.Hamiltonian:
            product_term = cuso.tensor_product(*cusp_terms, coeff=coeff)
            hamiltonian = cuso.Operator(hilbert_space_dims, (product_term, ))
            return hamiltonian
        else:
            d1_terms = []
            for term in cusp_terms:
                op_mat, degrees = term
                d1_terms.append((op_mat, degrees, (False,)))
            for term in reversed(cusp_terms):
                op_mat, degrees = term
                d1_terms.append((np.conj(op_mat).T, degrees, (True,)))
            D1 = cuso.tensor_product(*d1_terms, coeff = np.absolute(coeff) ** 2)

            d2_terms = []
            for term in cusp_terms:
                op_mat, degrees = term
                d2_terms.append((op_mat, degrees, (True,)))
            for term in reversed(cusp_terms):
                op_mat, degrees = term
                d2_terms.append((np.conj(op_mat).T, degrees, (True,)))

            D2 = cuso.tensor_product(*d2_terms, coeff = -0.5 * np.absolute(coeff) ** 2)
            d3_terms = []
            for term in cusp_terms:
                op_mat, degrees = term
                d3_terms.append((op_mat, degrees, (False,)))
            for term in reversed(cusp_terms):
                op_mat, degrees = term
                d3_terms.append((np.conj(op_mat).T, degrees, (False,)))
            D3 = cuso.tensor_product(*d3_terms, coeff = -0.5 * np.absolute(coeff) ** 2)
            lindblad = cuso.Operator(hilbert_space_dims, (D1, 1.0), (D2, 1.0), (D3, 1.0))
            return lindblad
    elif type(operator) == OperatorSum:
        if mode == OpConversionMode.Hamiltonian:
            product_terms = []
            for term in operator._terms:
                # print(f"Term: {term}, type: {type(term)}")
                if type(term) == ProductOperator:
                    cusp_terms = []
                    coeff = 1.0
                    for sub_term in term._operators:
                        # print(f"Sub-term: {sub_term}, type: {type(sub_term)}")
                        if type(sub_term) == ScalarOperator:
                            if sub_term._constant_value is None:
                                raise NotImplementedError("TODO: implement ScalarOperator with parameters")
                            coeff *= sub_term._constant_value
                        elif type(sub_term) == ElementaryOperator:
                            op_mat = cuso.optimize_strides(sub_term.to_matrix(dimensions))
                            # print(f"Op mat: {op_mat.shape}")
                            # print(f"Degree = {sub_term.degrees}")
                            cusp_terms.append(tuple((op_mat, sub_term.degrees)))
                        else:
                            raise NotImplementedError(f"Unsupported operator type: {type(sub_term)}")
                    product_term = cuso.tensor_product(*cusp_terms, coeff=coeff)
                    product_terms.append(product_term)
                else:
                    raise NotImplementedError(f"Unsupported operator type: {type(term)}")
            
            sum_of_products = product_terms[0]
            for i in range(1, len(product_terms)):
                sum_of_products += product_terms[i]
            hamiltonian = cuso.Operator(hilbert_space_dims, (sum_of_products, ))
            return hamiltonian
        else:
            # L = operator.to_matrix(dimensions)
            # # print(f"L mat: {l_mat}")
            # L_dag = np.conj(L).T
            # L_dag_L = np.dot(L_dag, L)
            # D1 = cuso.tensor_product(   # an operator term composed of a single elementary tensor operator
            #     (
            #         L,        # elementary tensor operator
            #         (0,),     # quantum degrees of freedom it acts on
            #         (False,)  # operator action duality (side: left/right) for each quantum degree of freedom
            #     ),
            #     (
            #         L_dag,        # elementary tensor operator
            #         (0,),     # quantum degrees of freedom it acts on
            #         (True,)  # operator action duality (side: left/right) for each quantum degree of freedom
            #     ),
            #     coeff = 1.0   # constant (static) coefficient
            # )


            # D2 = cuso.tensor_product(   # an operator term composed of a single elementary tensor operator
            #     (
            #         L_dag,        # elementary tensor operator
            #         (0,),     # quantum degrees of freedom it acts on
            #         (True,)  # operator action duality (side: left/right) for each quantum degree of freedom
            #     ),
            #     (
            #         L,        # elementary tensor operator
            #         (0,),     # quantum degrees of freedom it acts on
            #         (True,)  # operator action duality (side: left/right) for each quantum degree of freedom
            #     ),
            #     coeff = -0.5  # constant (static) coefficient
            # )

            # D3 = cuso.tensor_product(   # an operator term composed of a single elementary tensor operator
            #     (
            #         L,        # elementary tensor operator
            #         (0,),     # quantum degrees of freedom it acts on
            #         (False,)  # operator action duality (side: left/right) for each quantum degree of freedom
            #     ),
            #     (
            #         L_dag,        # elementary tensor operator
            #         (0,),     # quantum degrees of freedom it acts on
            #         (False,)  # operator action duality (side: left/right) for each quantum degree of freedom
            #     ),
            #     coeff = -0.5  # constant (static) coefficient
            # )

            # lindblad = cuso.Operator(hilbert_space_dims, (D1, 1.0), (D2, 1.0), (D3, 1.0))
            # return lindblad
            product_terms = []
            for term in operator._terms:
                # print(f"Term: {term}, type: {type(term)}")
                if type(term) == ProductOperator:
                    cusp_terms = []
                    coeff = 1.0
                    for sub_term in term._operators:
                        # print(f"Sub-term: {sub_term}, type: {type(sub_term)}")
                        if type(sub_term) == ScalarOperator:
                            if sub_term._constant_value is None:
                                raise NotImplementedError("TODO: implement ScalarOperator with parameters")
                            coeff *= sub_term._constant_value
                        elif type(sub_term) == ElementaryOperator:
                            op_mat = cuso.optimize_strides(sub_term.to_matrix(dimensions))
                            # print(f"Op mat: {op_mat.shape}")
                            # print(f"Degree = {sub_term.degrees}")
                            cusp_terms.append(tuple((op_mat, sub_term.degrees)))
                        else:
                            raise NotImplementedError(f"Unsupported operator type: {type(sub_term)}")
                    product_terms.append(tuple((cusp_terms, coeff)))
                else:
                    raise NotImplementedError(f"Unsupported operator type: {type(term)}")
            
            D_terms = []
            for lhs in product_terms:
                for rhs in product_terms:
                    d1_terms = []
                    for term in lhs[0]:
                        op_mat, degrees = term
                        d1_terms.append((op_mat, degrees, (False,)))
                    for term in reversed(rhs[0]):
                        op_mat, degrees = term
                        d1_terms.append((np.conj(op_mat).T, degrees, (True,)))
                    D1 = cuso.tensor_product(*d1_terms, coeff = lhs[1] * np.conj(rhs[1]))
                    D_terms.append(tuple((D1, 1.0)))

                    d2_terms = []
                    for term in reversed(rhs[0]):
                        op_mat, degrees = term
                        d2_terms.append((np.conj(op_mat).T, degrees, (True,)))
                    for term in lhs[0]:
                        op_mat, degrees = term
                        d2_terms.append((op_mat, degrees, (True,)))
                    

                    D2 = cuso.tensor_product(*d2_terms, coeff = -0.5 * lhs[1] * np.conj(rhs[1]))
                    D_terms.append(tuple((D2, 1.0)))

                    d3_terms = []
                    for term in lhs[0]:
                        op_mat, degrees = term
                        d3_terms.append((op_mat, degrees, (False,)))
                    for term in reversed(rhs[0]):
                        op_mat, degrees = term
                        d3_terms.append((np.conj(op_mat).T, degrees, (False,)))
                    D3 = cuso.tensor_product(*d3_terms, coeff = -0.5 * lhs[1] * np.conj(rhs[1]))
                    D_terms.append(tuple((D3, 1.0)))
            lindblad = cuso.Operator(hilbert_space_dims, *D_terms)
            return lindblad

    raise NotImplementedError(f"TODO: Implementation for this type: {type(operator)}")

# Abstract interface for an integrator
class Integrator:
    def __init__(self, liouvillian: cuso.Operator, ctx: cuso.WorkStream, n_steps: 1):
        self.state = None
        self.ctx = ctx
        self.n_steps = n_steps
        self.liouvillian = liouvillian
        self.liouvillian_action = None
    def set_state(self, state, t = 0.0):
        self.state = state
        self.t = t

    # Evolve to t
    def integrate(self, t):
        raise NotImplementedError

# First order integrator    
class EulerIntegrator(Integrator):
    
    def integrate(self, t):
        # print("Integrate to", t)
        if t <= self.t:
            raise ValueError("Integration time must be greater than current time")
        dt = (t - self.t)/self.n_steps
        # prepare operator action on a mixed quantum state
        if self.liouvillian_action is None:
            self.liouvillian_action = cuso.OperatorAction(self.ctx, (self.liouvillian, ))
            self.liouvillian_action.prepare(self.ctx, (self.state ,)) 

        for i in range(self.n_steps):
            current_t = self.t + i * dt
            rho0p = cuso.DenseDensityMatrix(self.ctx, cp.zeros_like(self.state.storage))
            # compute the operator action on a given quantum state
            # print("Compute @ t=", current_t)
            self.liouvillian_action.compute(current_t,        # time value
                            (),   # user-defined parameters
                            (self.state,),    # input quantum state
                            rho0p       # output quantum state
            )
            rho0p.inplace_scale(dt)
            self.state.inplace_add(rho0p)
            assert np.isclose(self.state.trace(), 1.0), "Density matrix is not normalized"

        self.t = t

# Runge-Kutta integrator
class RungeKuttaIntegrator(Integrator):
    
    def integrate(self, t):
        if t <= self.t:
            raise ValueError("Integration time must be greater than current time")
        dt = (t - self.t)/self.n_steps
        # prepare operator action on a mixed quantum state
        if self.liouvillian_action is None:
            self.liouvillian_action = cuso.OperatorAction(self.ctx, (self.liouvillian, ))
            self.liouvillian_action.prepare(self.ctx, (self.state ,)) 

        for i in range(self.n_steps):
            current_t = self.t + i * dt
            k1 = cuso.DenseDensityMatrix(self.ctx, cp.zeros_like(self.state.storage))
            # compute the operator action on a given quantum state
            # print("Compute @ t=", current_t)
            self.liouvillian_action.compute(current_t,        # time value
                            (),   # user-defined parameters
                            (self.state,),    # input quantum state
                            k1       # output quantum state
            )

            rho_temp = cp.copy(self.state.storage)
            rho_temp += ((dt/2) * k1.storage)
            k2 = cuso.DenseDensityMatrix(self.ctx, cp.zeros_like(self.state.storage))
            self.liouvillian_action.compute(current_t + dt/2,        # time value
                            (),   # user-defined parameters
                            (cuso.DenseDensityMatrix(self.ctx, rho_temp),),    # input quantum state
                            k2       # output quantum state
            )

            rho_temp = cp.copy(self.state.storage)
            rho_temp += ((dt/2) * k2.storage)
            k3 = cuso.DenseDensityMatrix(self.ctx, cp.zeros_like(self.state.storage))
            self.liouvillian_action.compute(current_t + dt/2,        # time value
                            (),   # user-defined parameters
                            (cuso.DenseDensityMatrix(self.ctx, rho_temp),),    # input quantum state
                            k3       # output quantum state
            )
            
            rho_temp = cp.copy(self.state.storage)
            rho_temp += ((dt) * k3.storage)
            k4 = cuso.DenseDensityMatrix(self.ctx, cp.zeros_like(self.state.storage))
            self.liouvillian_action.compute(current_t + dt,        # time value
                            (),   # user-defined parameters
                            (cuso.DenseDensityMatrix(self.ctx, rho_temp),),    # input quantum state
                            k4       # output quantum state
            )
            
            k1.inplace_scale(dt/6)
            k2.inplace_scale(dt/3)
            k3.inplace_scale(dt/3)
            k4.inplace_scale(dt/6)
            
            self.state.inplace_add(k1)
            self.state.inplace_add(k2)
            self.state.inplace_add(k3)
            self.state.inplace_add(k4)
            # assert np.isclose(self.state.trace(), 1.0), "Density matrix is not normalized"

        self.t = t

# [TEMP-CODE] Class to hold results in Python
class EvolveResult:
    def __init__(self):
        # list of arrays of expectation values
        # same order as observables
        self.expect = []
    def add_expectation(self, exp_array: List[float]):
        self.expect.append(exp_array)

# Master-equation solver using cuSuperOp
def evolve_me(hamiltonian: Operator, 
           dimensions: Mapping[int, int], 
           schedule: Schedule,
           initial_state: cudaq_runtime.State | Sequence[cudaq_runtime.States] | cuso.State,
           collapse_operators: Sequence[Operator] = [],
           observables: Sequence[Operator] = [], 
           store_intermediate_results = False) -> cudaq_runtime.EvolveResult | Sequence[cudaq_runtime.EvolveResult] | EvolveResult:
    # Conversion of operators to cuSuperOp
    ham_cusp = to_cusp_operator(hamiltonian, dimensions)

    hilbert_space_dims = tuple(dimensions[d] for d in range(len(dimensions)))
    ham_term = hamiltonian._evaluate(CuSuperOpHamConversion(dimensions))
    linblad_terms = []
    for c_op in collapse_operators:
        linblad_terms.append(c_op._evaluate(CuSuperOpHamConversion(dimensions)))
    liouvillian = constructLiouvillian(hilbert_space_dims, ham_term, linblad_terms)

    # hamiltonian = ham_cusp * (-1j)
    # liouvillian = hamiltonian - hamiltonian.dual() 
    # for op in collapse_operators:
    #     l_op = to_cusp_operator(op, dimensions, mode=OpConversionMode.Lindbladian)
    #     liouvillian += l_op
    
    # Note: we would need a CUDAQ state implementation for cuSuperOp
    if not isinstance(initial_state, cuso.State):
        raise NotImplementedError("TODO: list of input states")
    
    cuso_ctx = initial_state._ctx
    # FIXME: allow customization (select the integrator)
    # integrator = EulerIntegrator(ham_cusp, cuso_ctx, n_steps=100)
    integrator = RungeKuttaIntegrator(liouvillian, cuso_ctx, n_steps=10)
    expectation_op = [to_cusp_operator(observable, dimensions) for observable in observables]
    integrator.set_state(initial_state, schedule._steps[0])
    exp_vals = [[] for _ in observables]
    
    for step_idx, parameters in enumerate(schedule):
        # print(f"Current time = {schedule.current_step}")
        if step_idx > 0:
            integrator.integrate(schedule.current_step)
        for obs_idx, obs in enumerate(expectation_op):
            obs.prepare_expectation(cuso_ctx, integrator.state)
            exp_val = obs.compute_expectation(schedule.current_step, (), integrator.state)
            # print(f"Time = {schedule.current_step}: Exp = {float(cp.real(exp_val[0]))}")
            exp_vals[obs_idx].append(float(cp.real(exp_val[0])))
    
    result = EvolveResult()
    for exp_array in exp_vals:
        result.add_expectation(exp_array)
    return result

