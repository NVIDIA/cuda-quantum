# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from typing import Sequence, Mapping, List, Optional

from .cuso_helpers import CuSuperOpHamConversion, constructLiouvillian
from ..runtime.observe import observe
from .schedule import Schedule
from .expressions import Operator
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
from .cuso_helpers import cuso
from .cuso_state import CuSuperOpState, as_cuso_state
from .integrator import BaseIntegrator
from .builtin_integrators import RungeKuttaIntegrator, cuSuperOpTimeStepper
import cupy
import math
from .timing_helper import ScopeTimer


# Master-equation solver using `cuSuperOp`
def evolve_dynamics(
        hamiltonian: Operator,
        dimensions: Mapping[int, int],
        schedule: Schedule,
        initial_state: cudaq_runtime.State,
        collapse_operators: Sequence[Operator] = [],
        observables: Sequence[Operator] = [],
        store_intermediate_results=False,
        integrator: Optional[BaseIntegrator] = None
) -> cudaq_runtime.EvolveResult:
    if cuso is None:
        raise ImportError(
            "[nvidia-dynamics] Failed to import cuSuperOp module. Please check your installation."
        )

    # Reset the schedule
    schedule.reset()
    hilbert_space_dims = tuple(dimensions[d] for d in range(len(dimensions)))

    with ScopeTimer("evolve.as_cuso_state") as timer:
        initial_state = as_cuso_state(initial_state)

    if not isinstance(initial_state, CuSuperOpState):
        raise ValueError("Unknown type")

    if not initial_state.is_initialized():
        with ScopeTimer("evolve.init_state") as timer:
            initial_state.init_state(hilbert_space_dims)

    is_density_matrix = initial_state.is_density_matrix()
    me_solve = False
    if not is_density_matrix:
        if len(collapse_operators) == 0:
            me_solve = False
        else:
            with ScopeTimer("evolve.initial_state.to_dm") as timer:
                initial_state = initial_state.to_dm()
            me_solve = True
    else:
        # Always solve the master equation if the input is a density matrix
        me_solve = True

    with ScopeTimer("evolve.hamiltonian._evaluate") as timer:
        ham_term = hamiltonian._evaluate(
            CuSuperOpHamConversion(dimensions, schedule))
    linblad_terms = []
    for c_op in collapse_operators:
        with ScopeTimer("evolve.collapse_operators._evaluate") as timer:
            linblad_terms.append(
                c_op._evaluate(CuSuperOpHamConversion(dimensions, schedule)))

    with ScopeTimer("evolve.constructLiouvillian") as timer:
        liouvillian = constructLiouvillian(hilbert_space_dims, ham_term,
                                           linblad_terms, me_solve)

    initial_state = initial_state.get_impl()
    cuso_ctx = initial_state._ctx
    stepper = cuSuperOpTimeStepper(liouvillian, cuso_ctx)
    if integrator is None:
        integrator = RungeKuttaIntegrator(stepper)
    else:
        integrator.set_system(dimensions, schedule, hamiltonian,
                              collapse_operators)
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
        if step_idx > 0:
            with ScopeTimer("evolve.integrator.integrate") as timer:
                integrator.integrate(schedule.current_step)
        step_exp_vals = []
        for obs_idx, obs in enumerate(expectation_op):
            _, state = integrator.get_state()
            with ScopeTimer("evolve.prepare_expectation") as timer:
                obs.prepare_expectation(cuso_ctx, state)
            with ScopeTimer("evolve.compute_expectation") as timer:
                exp_val = obs.compute_expectation(schedule.current_step, (),
                                                  state)
            step_exp_vals.append(float(cupy.real(exp_val[0])))
        exp_vals.append(step_exp_vals)
        if store_intermediate_results:
            _, state = integrator.get_state()
            state_length = state.storage.size
            if is_density_matrix:
                dimension = int(math.sqrt(state_length))
                with ScopeTimer("evolve.intermediate_states.append") as timer:
                    intermediate_states.append(
                        cudaq_runtime.State.from_data(
                            state.storage.reshape((dimension, dimension))))
            else:
                dimension = state_length
                with ScopeTimer("evolve.intermediate_states.append") as timer:
                    intermediate_states.append(
                        cudaq_runtime.State.from_data(
                            state.storage.reshape((dimension,))))

    if store_intermediate_results:
        return cudaq_runtime.EvolveResult(intermediate_states, exp_vals)
    else:
        _, state = integrator.get_state()
        state_length = state.storage.size

        if is_density_matrix:
            dimension = int(math.sqrt(state_length))
            with ScopeTimer("evolve.final_state") as timer:
                final_state = cudaq_runtime.State.from_data(
                    state.storage.reshape((dimension, dimension)))
        else:
            dimension = state_length
            with ScopeTimer("evolve.final_state") as timer:
                final_state = cudaq_runtime.State.from_data(
                    state.storage.reshape((dimension,)))

        return cudaq_runtime.EvolveResult(final_state, exp_vals[-1])
