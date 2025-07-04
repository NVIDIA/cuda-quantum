# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
from typing import Sequence, Mapping, Optional

from .schedule import Schedule
from ..operators import Operator
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
from .helpers import InitialState, InitialStateArgT, IntermediateResultSave
from .integrator import BaseIntegrator
from .integrators.builtin_integrators import RungeKuttaIntegrator
from ..util.timing_helper import ScopeTimer
from . import nvqir_dynamics_bindings as bindings
from ..mlir._mlir_libs._quakeDialects.cudaq_runtime import MatrixOperator, SuperOperator


# Master-equation solver using `CuDensityMatState`
def evolve_dynamics(
    hamiltonian: Operator | SuperOperator | Sequence[Operator] |
    Sequence[SuperOperator],
    dimensions: Mapping[int, int],
    schedule: Schedule,
    initial_state: InitialStateArgT | Sequence[cudaq_runtime.State],
    collapse_operators: Sequence[Operator] | Sequence[Sequence[Operator]] = [],
    observables: Sequence[Operator] = [],
    store_intermediate_results: IntermediateResultSave = IntermediateResultSave.
    NONE,
    integrator: Optional[BaseIntegrator] = None,
    max_batch_size: Optional[int] = None
) -> cudaq_runtime.EvolveResult | Sequence[cudaq_runtime.EvolveResult]:
    # Reset the schedule
    schedule.reset()
    hilbert_space_dims = tuple(dimensions[d] for d in range(len(dimensions)))

    if not isinstance(store_intermediate_results, IntermediateResultSave):
        raise TypeError(
            "store_intermediate_results must be an instance of IntermediateResultSave"
        )
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

    has_collapse_operators = False

    if isinstance(hamiltonian, Sequence):
        # This is batched operators evolve.
        if len(collapse_operators) > 0:
            if not isinstance(collapse_operators[0], Sequence):
                raise ValueError(
                    "'collapse_operators' must be a sequence of sequences when supplying a sequence of Hamiltonians"
                )
            if len(hamiltonian) != len(collapse_operators):
                raise ValueError(
                    "Number of Hamiltonians and collapse operators must match")

        if len(initial_state) != len(hamiltonian):
            raise ValueError(
                "Number of initial states must match number of Hamiltonians")
        # Make sure all Hamiltonians are of the same type.
        if not all(isinstance(op, Operator) for op in hamiltonian) and not all(
                isinstance(op, SuperOperator) for op in hamiltonian):
            raise ValueError(
                "All Hamiltonians must be of the same type (either Operator or SuperOperator)"
            )
        isSuperOperator = isinstance(hamiltonian[0], SuperOperator)
        if isSuperOperator:
            if len(collapse_operators) > 0:
                raise ValueError(
                    "'collapse_operators' must be empty when supplying a sequence of super-operators"
                )

            can_be_batched = bindings.checkSuperOpBatchingCompatibility(
                hamiltonian)
            batch_size = len(hamiltonian)

            if not can_be_batched:
                if max_batch_size is not None and max_batch_size > 1:
                    raise ValueError(
                        f"The input super-operators are not compatible for batching. Unable to run batched simulation with the requested batch size {max_batch_size}."
                    )
                if max_batch_size is None:
                    print(
                        "Warning: The input super-operators are not compatible for batching. Running the simulation in non-batched mode."
                    )
                # Only run sequentially
                max_batch_size = 1
            else:
                if max_batch_size is not None and max_batch_size > batch_size:
                    raise ValueError(
                        f"Invalid max_batch_size {max_batch_size} for the given number of super-operators {batch_size}."
                    )
                if max_batch_size is not None and max_batch_size < 1:
                    raise ValueError(
                        f"Invalid max_batch_size {max_batch_size}. It must be at least 1."
                    )
            if max_batch_size is None:
                # Use the number of super-operators as the batch size.
                max_batch_size = batch_size

            if max_batch_size < batch_size:
                # Split the super-operators into batches.
                hamiltonian = [
                    hamiltonian[i:i + max_batch_size]
                    for i in range(0, len(hamiltonian), max_batch_size)
                ]

                initial_state = [
                    initial_state[i:i + max_batch_size]
                    for i in range(0, len(initial_state), max_batch_size)
                ]
                all_results = []
                for batch_idx in range(len(hamiltonian)):
                    batch_hamiltonian = hamiltonian[batch_idx] if len(
                        hamiltonian[batch_idx]
                    ) > 1 else hamiltonian[batch_idx][0]
                    batch_initial_state = initial_state[batch_idx] if len(
                        initial_state[batch_idx]
                    ) > 1 else initial_state[batch_idx][0]
                    # Recursively call evolve_dynamics for each batch.
                    result = evolve_dynamics(
                        batch_hamiltonian,
                        dimensions,
                        schedule,
                        batch_initial_state,
                        collapse_operators=[],
                        observables=observables,
                        store_intermediate_results=store_intermediate_results,
                        integrator=integrator,
                        max_batch_size=max_batch_size)
                    all_results.append(result)
                return all_results

            for super_op in hamiltonian:
                for (left_op, right_op) in super_op:
                    if right_op is not None:
                        has_collapse_operators = True
                        break
            integrator.set_system(dimensions, schedule, hamiltonian)
        else:
            for collapse_ops in collapse_operators:
                if len(collapse_ops) > 0:
                    has_collapse_operators = True
                    break
            collapse_operators = [[MatrixOperator(op)
                                   for op in collapse_ops]
                                  for collapse_ops in collapse_operators]
            hamiltonian = [MatrixOperator(op) for op in hamiltonian]

            can_be_batched = bindings.checkBatchingCompatibility(
                hamiltonian, collapse_operators)
            batch_size = len(hamiltonian)

            if not can_be_batched:
                if max_batch_size is not None and max_batch_size > 1:
                    raise ValueError(
                        f"The input Hamiltonian and collapse operators are not compatible for batching. Unable to run batched simulation with the requested batch size {max_batch_size}."
                    )
                if max_batch_size is None:
                    print(
                        "Warning: The input Hamiltonian and collapse operators are not compatible for batching. Running the simulation in non-batched mode."
                    )
                # Only run sequentially
                max_batch_size = 1
            else:
                if max_batch_size is not None and max_batch_size > batch_size:
                    raise ValueError(
                        f"Invalid max_batch_size {max_batch_size} for the given number of Hamiltonian operators {batch_size}."
                    )
                if max_batch_size is not None and max_batch_size < 1:
                    raise ValueError(
                        f"Invalid max_batch_size {max_batch_size}. It must be at least 1."
                    )
            if max_batch_size is None:
                # Use the number of Hamiltonian operators as the batch size.
                max_batch_size = batch_size

            if max_batch_size < batch_size:
                # Split the Hamiltonian into batches.
                hamiltonian = [
                    hamiltonian[i:i + max_batch_size]
                    for i in range(0, len(hamiltonian), max_batch_size)
                ]

                initial_state = [
                    initial_state[i:i + max_batch_size]
                    for i in range(0, len(initial_state), max_batch_size)
                ]

                if len(collapse_operators) > 0:
                    collapse_operators = [
                        collapse_operators[i:i + max_batch_size] for i in range(
                            0, len(collapse_operators), max_batch_size)
                    ]
                else:
                    # If no collapse operators are provided, use empty lists for each batch.
                    collapse_operators = [[]] * len(hamiltonian)

                all_results = []
                for batch_idx in range(len(hamiltonian)):
                    batch_hamiltonian = hamiltonian[batch_idx] if len(
                        hamiltonian[batch_idx]
                    ) > 1 else hamiltonian[batch_idx][0]
                    batch_initial_state = initial_state[batch_idx] if len(
                        initial_state[batch_idx]
                    ) > 1 else initial_state[batch_idx][0]
                    batch_collapse_operators = collapse_operators[
                        batch_idx] if len(
                            collapse_operators[batch_idx]
                        ) > 1 else collapse_operators[batch_idx][0]
                    # Recursively call evolve_dynamics for each batch.
                    result = evolve_dynamics(
                        batch_hamiltonian,
                        dimensions,
                        schedule,
                        batch_initial_state,
                        collapse_operators=batch_collapse_operators,
                        observables=observables,
                        store_intermediate_results=store_intermediate_results,
                        integrator=integrator,
                        max_batch_size=max_batch_size)
                    all_results.append(result)
                return all_results

            integrator.set_system(dimensions, schedule, hamiltonian,
                                  collapse_operators)
    else:
        if isinstance(hamiltonian, SuperOperator):
            if len(collapse_operators) > 0:
                raise ValueError(
                    "'collapse_operators' must be empty when supplying the super-operator"
                )
            integrator.set_system(dimensions, schedule, hamiltonian)
            for (left_op, right_op) in hamiltonian:
                if right_op is not None:
                    has_collapse_operators = True
        else:
            has_collapse_operators = len(collapse_operators) > 0
            collapse_operators = [
                MatrixOperator(op) for op in collapse_operators
            ]
            integrator.set_system(dimensions, schedule,
                                  MatrixOperator(hamiltonian),
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
                                                    has_collapse_operators)
        is_batched_evolve = True
    else:
        if isinstance(initial_state, InitialState):
            initial_state = bindings.createInitialState(initial_state,
                                                        dimensions,
                                                        has_collapse_operators)
        else:
            initial_state = bindings.initializeState(initial_state,
                                                     hilbert_space_dims_list,
                                                     has_collapse_operators, 1)
    integrator.set_state(initial_state, schedule._steps[0])

    exp_vals = [[] for _ in range(batch_size)]
    intermediate_states = [[] for _ in range(batch_size)]
    for step_idx, parameters in enumerate(schedule):
        if step_idx > 0:
            with ScopeTimer("evolve.integrator.integrate") as timer:
                integrator.integrate(schedule.current_step)
        # If we store intermediate values, compute them for each step.
        # Otherwise, just for the last step.
        if store_intermediate_results != IntermediateResultSave.NONE or step_idx == (
                len(schedule) - 1):
            step_exp_vals = [[] for _ in range(batch_size)]
            _, state = integrator.get_state()
            for obs_idx, obs in enumerate(expectation_op):
                obs.prepare(state)
                exp_val = obs.compute(state, schedule.current_step)
                for i in range(batch_size):
                    step_exp_vals[i].append(exp_val[i])

            for i in range(batch_size):
                exp_vals[i].append(step_exp_vals[i])
            # Store all intermediate states if requested. Otherwise, only the last state.
            if store_intermediate_results == IntermediateResultSave.ALL or step_idx == (
                    len(schedule) - 1):
                split_states = bindings.splitBatchedState(state)
                for i in range(batch_size):
                    intermediate_states[i].append(split_states[i])

    bindings.clearContext()
    results = [
        cudaq_runtime.EvolveResult(state, exp_val)
        for state, exp_val in zip(intermediate_states, exp_vals)
    ] if (store_intermediate_results != IntermediateResultSave.NONE) else [
        cudaq_runtime.EvolveResult(state[-1], exp_val[-1])
        for state, exp_val in zip(intermediate_states, exp_vals)
    ]

    if is_batched_evolve:
        return results
    else:
        return results[0]
