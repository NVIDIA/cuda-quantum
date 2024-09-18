from __future__ import annotations
from typing import Sequence, Mapping, List, Optional

from .cuso_helpers import CuSuperOpHamConversion, constructLiouvillian
from ..runtime.observe import observe
from .schedule import Schedule
from .expressions import Operator
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
import cusuperop as cuso
from .cuso_state import CuSuperOpState
from .integrator import BaseIntegrator
from .builtin_integrators import RungeKuttaIntegrator, cuSuperOpTimeStepper
from .scipy_integrators import ScipyZvodeIntegrator
import cupy
import copy

# Master-equation solver using cuSuperOp
def evolve_me(hamiltonian: Operator, 
           dimensions: Mapping[int, int], 
           schedule: Schedule,
           initial_state: cudaq_runtime.State | Sequence[cudaq_runtime.State],
           collapse_operators: Sequence[Operator] = [],
           observables: Sequence[Operator] = [], 
           store_intermediate_results = False,
           integrator: Optional[BaseIntegrator] = None) -> cudaq_runtime.EvolveResult | Sequence[cudaq_runtime.EvolveResult]:
    hilbert_space_dims = tuple(dimensions[d] for d in range(len(dimensions)))
    ham_term = hamiltonian._evaluate(CuSuperOpHamConversion(dimensions))
    linblad_terms = []
    for c_op in collapse_operators:
        linblad_terms.append(c_op._evaluate(CuSuperOpHamConversion(dimensions)))
    liouvillian = constructLiouvillian(hilbert_space_dims, ham_term, linblad_terms)

    # Note: we would need a CUDAQ state implementation for cuSuperOp
    if not isinstance(initial_state, cudaq_runtime.State):
        raise NotImplementedError("TODO: list of input states")
    
    initial_state = initial_state.get_impl()

    if not isinstance(initial_state, CuSuperOpState):
        raise ValueError("Unknown type")

    initial_state = initial_state.get_impl()
    cuso_ctx = initial_state._ctx
    stepper = cuSuperOpTimeStepper(liouvillian, cuso_ctx)
    if integrator is None:
        integrator = RungeKuttaIntegrator(stepper)
    else:
        integrator.set_system(dimensions, hamiltonian, collapse_operators)
    expectation_op = [cuso.Operator(hilbert_space_dims, (observable._evaluate(CuSuperOpHamConversion(dimensions)), 1.0)) for observable in observables]
    integrator.set_state(initial_state, schedule._steps[0])
    exp_vals = []
    intermediate_states = []

    for step_idx, parameters in enumerate(schedule):
        # print(f"Current time = {schedule.current_step}")
        if step_idx > 0:
            integrator.integrate(schedule.current_step)
        step_exp_vals = []
        for obs_idx, obs in enumerate(expectation_op):
            _, state = integrator.get_state()
            obs.prepare_expectation(cuso_ctx, state)
            exp_val = obs.compute_expectation(schedule.current_step, (), state)
            # print(f"Time = {schedule.current_step}: Exp = {float(cp.real(exp_val[0]))}")
            step_exp_vals.append(float(cupy.real(exp_val[0])))
        exp_vals.append(step_exp_vals)
        if store_intermediate_results:
            _, state = integrator.get_state()
            intermediate_states.append(cudaq_runtime.State.wrap_py_state(CuSuperOpState(copy.deepcopy(state.storage))))

    if store_intermediate_results:
        return cudaq_runtime.EvolveResult(intermediate_states, exp_vals)
    else:
        _, final_cuso_state = integrator.get_state()
        final_state = cudaq_runtime.State.wrap_py_state(CuSuperOpState(copy.deepcopy(final_cuso_state.storage)))
        return cudaq_runtime.EvolveResult(final_state, exp_vals[-1])

