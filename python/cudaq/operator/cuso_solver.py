from __future__ import annotations
from typing import Sequence, Mapping, List

from .cuso_helpers import CuSuperOpHamConversion, constructLiouvillian
from ..runtime.observe import observe
from .schedule import Schedule
from .expressions import Operator
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
import cusuperop as cuso

from .builtin_integrators import RungeKuttaIntegrator, cuSuperOpTimeStepper
import cupy

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
    hilbert_space_dims = tuple(dimensions[d] for d in range(len(dimensions)))
    ham_term = hamiltonian._evaluate(CuSuperOpHamConversion(dimensions))
    linblad_terms = []
    for c_op in collapse_operators:
        linblad_terms.append(c_op._evaluate(CuSuperOpHamConversion(dimensions)))
    liouvillian = constructLiouvillian(hilbert_space_dims, ham_term, linblad_terms)

    # Note: we would need a CUDAQ state implementation for cuSuperOp
    if not isinstance(initial_state, cuso.State):
        raise NotImplementedError("TODO: list of input states")
    
    cuso_ctx = initial_state._ctx
    # FIXME: allow customization (select the integrator)
    stepper = cuSuperOpTimeStepper(liouvillian, cuso_ctx)
    integrator = RungeKuttaIntegrator(stepper, nsteps=10)
    expectation_op = [cuso.Operator(hilbert_space_dims, (observable._evaluate(CuSuperOpHamConversion(dimensions)), 1.0)) for observable in observables]
    integrator.set_state(initial_state, schedule._steps[0])
    exp_vals = [[] for _ in observables]
    
    for step_idx, parameters in enumerate(schedule):
        # print(f"Current time = {schedule.current_step}")
        if step_idx > 0:
            integrator.integrate(schedule.current_step)
        for obs_idx, obs in enumerate(expectation_op):
            _, state = integrator.get_state()
            obs.prepare_expectation(cuso_ctx, state)
            exp_val = obs.compute_expectation(schedule.current_step, (), state)
            # print(f"Time = {schedule.current_step}: Exp = {float(cp.real(exp_val[0]))}")
            exp_vals[obs_idx].append(float(cupy.real(exp_val[0])))
    
    result = EvolveResult()
    for exp_array in exp_vals:
        result.add_expectation(exp_array)
    return result

