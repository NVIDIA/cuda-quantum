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
import math
from cupy.cuda.memory import MemoryPointer, UnownedMemory

def as_cuso_state(state):
    tensor = state.getTensor()
    pDevice  = tensor.data()
    dtype = cupy.complex128
    # print(f"Cupy pointer: {hex(pDevice)}")
    sizeByte = tensor.get_num_elements() *  tensor.get_element_size()
    mem = UnownedMemory(pDevice, sizeByte, owner = state)
    memptr = MemoryPointer(mem, 0)
    cupy_array = cupy.ndarray(tensor.get_num_elements(), dtype=dtype, memptr=memptr)
    return CuSuperOpState(cupy_array)

# Master-equation solver using cuSuperOp
def evolve_me(
    hamiltonian: Operator,
    dimensions: Mapping[int, int],
    schedule: Schedule,
    initial_state: cudaq_runtime.State | Sequence[cudaq_runtime.State],
    collapse_operators: Sequence[Operator] = [],
    observables: Sequence[Operator] = [],
    store_intermediate_results=False,
    integrator: Optional[BaseIntegrator] = None
) -> cudaq_runtime.EvolveResult | Sequence[cudaq_runtime.EvolveResult]:
    # Reset the schedule
    schedule.reset()
    hilbert_space_dims = tuple(dimensions[d] for d in range(len(dimensions)))

    # Note: we would need a CUDAQ state implementation for cuSuperOp
    if not isinstance(initial_state, cudaq_runtime.State):
        raise NotImplementedError("TODO: list of input states")
    initial_state = as_cuso_state(initial_state)

    if not isinstance(initial_state, CuSuperOpState):
        raise ValueError("Unknown type")

    if not initial_state.is_initialized():
        initial_state.init_state(hilbert_space_dims)

    is_density_matrix = initial_state.is_density_matrix()
    me_solve = False
    if not is_density_matrix:
        if len(collapse_operators) == 0:
            me_solve = False
        else:
            initial_state = initial_state.to_dm()
            me_solve = True
    else:
        # Always solve the master equation if the input is a density matrix
        me_solve = True

    ham_term = hamiltonian._evaluate(CuSuperOpHamConversion(dimensions))
    linblad_terms = []
    for c_op in collapse_operators:
        linblad_terms.append(c_op._evaluate(CuSuperOpHamConversion(dimensions)))
    liouvillian = constructLiouvillian(hilbert_space_dims, ham_term,
                                       linblad_terms, me_solve)

    initial_state = initial_state.get_impl()
    cuso_ctx = initial_state._ctx
    stepper = cuSuperOpTimeStepper(liouvillian, cuso_ctx)
    if integrator is None:
        integrator = RungeKuttaIntegrator(stepper)
    else:
        integrator.set_system(dimensions, hamiltonian, collapse_operators)
    expectation_op = [
        cuso.Operator(
            hilbert_space_dims,
            (observable._evaluate(CuSuperOpHamConversion(dimensions)), 1.0))
        for observable in observables
    ]
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
            state_length = state.storage.size
            if is_density_matrix:
                dimension = int(math.sqrt(state_length))
                intermediate_states.append(cudaq_runtime.State.from_data(state.storage.reshape((dimension, dimension))))
            else:
                dimension = state_length
                intermediate_states.append(cudaq_runtime.State.from_data(state.storage.reshape((dimension,))))


    if store_intermediate_results:
        return cudaq_runtime.EvolveResult(intermediate_states, exp_vals)
    else:
        _, state = integrator.get_state()
        state_length = state.storage.size
        
        if is_density_matrix:
            dimension = int(math.sqrt(state_length))
            final_state = cudaq_runtime.State.from_data(state.storage.reshape((dimension, dimension)))
        else:
            dimension = state_length
            final_state = cudaq_runtime.State.from_data(state.storage.reshape((dimension,)))
                
        return cudaq_runtime.EvolveResult(final_state, exp_vals[-1])
