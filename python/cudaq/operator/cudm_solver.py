# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from typing import Sequence, Mapping, List, Optional

from .cudm_helpers import CuDensityMatOpConversion, constructLiouvillian
from ..runtime.observe import observe
from .schedule import Schedule
from .expressions import Operator
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
from .cudm_helpers import cudm, CudmStateType
from .cudm_state import CuDensityMatState, as_cudm_state
from .helpers import InitialState, InitialStateArgT
from .integrator import BaseIntegrator
from .integrators.builtin_integrators import RungeKuttaIntegrator, cuDensityMatTimeStepper
import cupy
import math
from ..util.timing_helper import ScopeTimer


# Master-equation solver using `CuDensityMatState`
def evolve_dynamics(
        hamiltonian: Operator,
        dimensions: Mapping[int, int],
        schedule: Schedule,
        initial_state: InitialStateArgT,
        collapse_operators: Sequence[Operator] = [],
        observables: Sequence[Operator] = [],
        store_intermediate_results=False,
        integrator: Optional[BaseIntegrator] = None
) -> cudaq_runtime.EvolveResult:
    if cudm is None:
        raise ImportError(
            "[dynamics target] Failed to import cuquantum density module. Please check your installation."
        )

    # Reset the schedule
    schedule.reset()
    hilbert_space_dims = tuple(dimensions[d] for d in range(len(dimensions)))

    # Check that the integrator can support distributed state if this is a distributed simulation.
    if cudaq_runtime.mpi.is_initialized() and cudaq_runtime.mpi.num_ranks(
    ) > 1 and integrator is not None and not integrator.support_distributed_state(
    ):
        raise ValueError(
            f"Integrator {type(integrator).__name__} does not support distributed state."
        )

    if isinstance(initial_state, InitialState):
        has_collapse_operators = len(collapse_operators) > 0
        initial_state = CuDensityMatState.create_initial_state(
            initial_state, hilbert_space_dims, has_collapse_operators)
    else:
        with ScopeTimer("evolve.as_cudm_state") as timer:
            initial_state = as_cudm_state(initial_state)

    if not isinstance(initial_state, CuDensityMatState):
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
            CuDensityMatOpConversion(dimensions, schedule))
    linblad_terms = []
    for c_op in collapse_operators:
        with ScopeTimer("evolve.collapse_operators._evaluate") as timer:
            linblad_terms.append(
                c_op._evaluate(CuDensityMatOpConversion(dimensions, schedule)))

    with ScopeTimer("evolve.constructLiouvillian") as timer:
        liouvillian = constructLiouvillian(hilbert_space_dims, ham_term,
                                           linblad_terms, me_solve)

    initial_state = initial_state.get_impl()
    cudm_ctx = initial_state._ctx
    stepper = cuDensityMatTimeStepper(liouvillian, cudm_ctx)
    if integrator is None:
        integrator = RungeKuttaIntegrator(stepper)
    else:
        integrator.set_system(dimensions, schedule, hamiltonian,
                              collapse_operators)
    expectation_op = [
        cudm.Operator(
            hilbert_space_dims,
            (observable._evaluate(CuDensityMatOpConversion(dimensions)), 1.0))
        for observable in observables
    ]
    integrator.set_state(initial_state, schedule._steps[0])
    exp_vals = []
    intermediate_states = []
    for step_idx, parameters in enumerate(schedule):
        if step_idx > 0:
            with ScopeTimer("evolve.integrator.integrate") as timer:
                integrator.integrate(schedule.current_step)
        # If we store intermediate values, compute them for each step.
        # Otherwise, just for the last step.
        if store_intermediate_results or step_idx == (len(schedule) - 1):
            step_exp_vals = []
            for obs_idx, obs in enumerate(expectation_op):
                _, state = integrator.get_state()
                with ScopeTimer("evolve.prepare_expectation") as timer:
                    obs.prepare_expectation(cudm_ctx, state)
                with ScopeTimer("evolve.compute_expectation") as timer:
                    exp_val = obs.compute_expectation(schedule.current_step, (),
                                                      state)
                step_exp_vals.append(float(cupy.real(exp_val[0])))
            exp_vals.append(step_exp_vals)
        if store_intermediate_results:
            _, state = integrator.get_state()
            state_length = state.storage.size
            if is_density_matrix and not CuDensityMatState.is_multi_process():
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

        # Only reshape the data into a density matrix is this is a single-GPU state.
        # In a multi-GPU state, the density matrix is sliced, hence we cannot reshape each slice into a density matrix form.
        # The data is returned as a flat buffer in this case.
        if is_density_matrix and not CuDensityMatState.is_multi_process():
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
