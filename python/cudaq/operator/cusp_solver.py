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

import cupy as cp
import numpy as np

import cusuperop as cuso

from time import sleep

# Helper to convert to cuSuperOp format
def to_cusp_operator(operator: Operator, dimensions: Mapping[int, int]) -> cuso.Operator:
    hilbert_space_dims = tuple(dimensions[d] for d in range(len(dimensions)))
    # print(f"Op: {operator}, type: {type(operator)}")
    if type(operator) == ElementaryOperator:
        ham_term = cuso.OperatorTerm()
        if len(operator.parameters) > 0:
            raise NotImplementedError("TODO: implement ElementaryOperator with parameters")
        op_mat = operator.to_matrix(dimensions)
        ham_term.append([cuso.GeneralOperator(cuso.CallbackTensor(op_mat), operator.degrees, ((False,) * len(operator.degrees)))], 1.0)
        hamiltonian = cuso.Operator(hilbert_space_dims, (ham_term, 1.0))
        return hamiltonian
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
        product_term = cuso.tensor_product(*cusp_terms, coeff=coeff)
        hamiltonian = cuso.Operator(hilbert_space_dims, (product_term, ))
        return hamiltonian
    elif type(operator) == OperatorSum:
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

    raise NotImplementedError(f"TODO: Implementation for this type: {type(operator)}")

# Abstract interface for an integrator
class Integrator:
    def __init__(self, ham: cuso.Operator, ctx: cuso.WorkStream, n_steps: 1):
        self.ham = ham
        self.state = None
        self.ctx = ctx
        self.n_steps = n_steps
        self.liouvillian = None
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
        if self.liouvillian is None:
            hamiltonian = self.ham * (-1j * dt)
            self.liouvillian = hamiltonian - hamiltonian.dual() 
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
            self.state.inplace_add(rho0p)
            assert np.isclose(self.state.trace(), 1.0)

        self.t = t

# Runge-Kutta integrator
class RungeKuttaIntegrator(Integrator):
    
    def integrate(self, t):
        if t <= self.t:
            raise ValueError("Integration time must be greater than current time")
        dt = (t - self.t)/self.n_steps
        # prepare operator action on a mixed quantum state
        if self.liouvillian is None:
            hamiltonian = self.ham * (-1j)
            self.liouvillian = hamiltonian - hamiltonian.dual() 
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
            assert np.isclose(self.state.trace(), 1.0)

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
    
    if len(collapse_operators) > 0:
        raise NotImplementedError("TODO: Implement collapse operators")
    
    # Note: we would need a CUDAQ state implementation for cuSuperOp
    if not isinstance(initial_state, cuso.State):
        raise NotImplementedError("TODO: list of input states")
    
    cuso_ctx = initial_state._ctx
    # FIXME: allow customization (select the integrator)
    # integrator = EulerIntegrator(ham_cusp, cuso_ctx, n_steps=100)
    integrator = RungeKuttaIntegrator(ham_cusp, cuso_ctx, n_steps=10)
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

