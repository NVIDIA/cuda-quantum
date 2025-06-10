# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from typing import Sequence, Mapping, Optional

from ..runtime.observe import observe
from .schedule import Schedule
from ..operators import Operator
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
from .helpers import InitialState, InitialStateArgT
from .integrator import BaseIntegrator
from .integrators.builtin_integrators import RungeKuttaIntegrator
from ..util.timing_helper import ScopeTimer
from . import nvqir_dynamics_bindings as bindings
from ..mlir._mlir_libs._quakeDialects.cudaq_runtime import MatrixOperator


# Master-equation solver using `CuDensityMatState`
def evolve_dynamics(
    hamiltonian: Operator,
    dimensions: Mapping[int, int],
    schedule: Schedule,
    initial_state: InitialStateArgT | Sequence[cudaq_runtime.State],
    collapse_operators: Sequence[Operator] = [],
    observables: Sequence[Operator] = [],
    store_intermediate_results=False,
    integrator: Optional[BaseIntegrator] = None
) -> cudaq_runtime.EvolveResult | Sequence[cudaq_runtime.EvolveResult]:
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

    if integrator is None:
        # Default integrator if not provided.
        integrator = RungeKuttaIntegrator()

    collapse_operators = [MatrixOperator(op) for op in collapse_operators]
    integrator.set_system(dimensions, schedule, MatrixOperator(hamiltonian),
                          collapse_operators)
    hilbert_space_dims_list = list(hilbert_space_dims)
    expectation_op = [
        bindings.CuDensityMatExpectation(MatrixOperator(observable),
                                         hilbert_space_dims_list)
        for observable in observables
    ]

    batch_size = 1
    is_batched_evolve = False
    if isinstance(initial_state, Sequence):
        batch_size = len(initial_state)
        initial_state = bindings.createBatchedState(initial_state,
                                                    hilbert_space_dims_list,
                                                    len(collapse_operators) > 0)
        is_batched_evolve = True
    else:
        if isinstance(initial_state, InitialState):
            has_collapse_operators = len(collapse_operators) > 0
            initial_state = bindings.createInitialState(initial_state,
                                                        dimensions,
                                                        has_collapse_operators)
        else:
            initial_state = bindings.initializeState(
                initial_state, hilbert_space_dims_list,
                len(collapse_operators) > 0, 1)
    integrator.set_state(initial_state, schedule._steps[0])

    exp_vals = [[] for _ in range(batch_size)]
    intermediate_states = [[] for _ in range(batch_size)]
    for step_idx, parameters in enumerate(schedule):
        if step_idx > 0:
            with ScopeTimer("evolve.integrator.integrate") as timer:
                integrator.integrate(schedule.current_step)
        # If we store intermediate values, compute them for each step.
        # Otherwise, just for the last step.
        if store_intermediate_results or step_idx == (len(schedule) - 1):
            step_exp_vals = [[] for _ in range(batch_size)]
            _, state = integrator.get_state()
            for obs_idx, obs in enumerate(expectation_op):
                obs.prepare(state)
                exp_val = obs.compute(state, schedule.current_step)
                for i in range(batch_size):
                    step_exp_vals[i].append(exp_val[i])
            split_states = bindings.splitBatchedState(state)
            for i in range(batch_size):
                exp_vals[i].append(step_exp_vals[i])
                intermediate_states[i].append(split_states[i])

    bindings.clearContext()
    results = [
        cudaq_runtime.EvolveResult(state, exp_val)
        for state, exp_val in zip(intermediate_states, exp_vals)
    ] if store_intermediate_results else [
        cudaq_runtime.EvolveResult(state[-1], exp_val[-1])
        for state, exp_val in zip(intermediate_states, exp_vals)
    ]

    if is_batched_evolve:
        return results
    else:
        return results[0]
