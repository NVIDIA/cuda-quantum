# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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


def should_use_mixed_state(
    hamiltonian: Operator | SuperOperator | Sequence[Operator] |
    Sequence[SuperOperator],
    collapse_operators: Sequence[Operator] | Sequence[Sequence[Operator]] = []
) -> bool:
    """
    Check whether the dynamics solver should use mixed states.
    i.e., promote the input state from a state vector to a density matrix if needed.
    """
    if len(collapse_operators) > 0:
        if isinstance(collapse_operators[0], Operator):
            return True
        elif isinstance(collapse_operators[0], Sequence):
            # Batch of collapse operators.
            # Check if any of the collapse operators are not empty.
            for collapse_ops in collapse_operators:
                if len(collapse_ops) > 0:
                    return True

    if isinstance(hamiltonian, Sequence):
        return any(isinstance(op, SuperOperator) for op in hamiltonian)

    # Helper to check if the super-operator has right apply operators.
    def has_right_apply(super_op: SuperOperator) -> bool:
        for (left_op, right_op) in super_op:
            if right_op is not None:
                return True
        return False

    if isinstance(hamiltonian, SuperOperator):
        # If the hamiltonian is a super-operator, check if it has right apply operators.
        return has_right_apply(hamiltonian)
    elif isinstance(hamiltonian, Sequence):
        # If the hamiltonian is a sequence of operators, check if any of them is a super-operator.
        return any(
            isinstance(op, SuperOperator) and has_right_apply(op)
            for op in hamiltonian)

    return False


def validate_evolve_dynamics_args(hamiltonian: Operator | SuperOperator |
                                  Sequence[Operator] | Sequence[SuperOperator],
                                  collapse_operators: Sequence[Operator] |
                                  Sequence[Sequence[Operator]]):
    """ 
    Validate the arguments for the `evolve_dynamics` function. 
    """
    if not isinstance(collapse_operators, Sequence):
        raise TypeError("'collapse_operators' must be an array.")

    if isinstance(hamiltonian, Sequence):
        # Batch simulation

        if len(hamiltonian) == 0:
            raise ValueError("Hamiltonian sequence cannot be empty")

        # Make sure all Hamiltonians are of the same type.
        if not all(isinstance(op, Operator) for op in hamiltonian) and not all(
                isinstance(op, SuperOperator) for op in hamiltonian):
            raise ValueError(
                "All Hamiltonians must be of the same type (either Operator or SuperOperator)"
            )
        if isinstance(hamiltonian[0], Operator):
            hamiltonian = [MatrixOperator(op) for op in hamiltonian]
            collapse_operators = [[MatrixOperator(op)
                                   for op in collapse_ops]
                                  for collapse_ops in collapse_operators]
    else:
        # Single Hamiltonian
        # Collapsed operators must be a single-level list of operators.
        if not all(isinstance(op, Operator) for op in collapse_operators):
            raise ValueError(
                "'collapse_operators' must be a sequence of Operators.")
        if isinstance(hamiltonian, Operator):
            # If the hamiltonian is a single operator, convert it to a MatrixOperator.
            hamiltonian = MatrixOperator(hamiltonian)
            collapse_operators = [
                MatrixOperator(op) for op in collapse_operators
            ]

    isSuperOperator = isinstance(hamiltonian, SuperOperator) or (isinstance(
        hamiltonian, Sequence) and isinstance(hamiltonian[0], SuperOperator))
    if isSuperOperator:
        # If the hamiltonian is a super-operator, collapse operators must be empty.
        if len(collapse_operators) > 0:
            raise ValueError(
                "'collapse_operators' must be empty when supplying a super-operator"
            )

    return hamiltonian, collapse_operators


def determine_batch_size(hamiltonians: Sequence[Operator] |
                         Sequence[SuperOperator],
                         collapse_operators: Sequence[Operator] |
                         Sequence[Sequence[Operator]],
                         max_batch_size: Optional[int] = None) -> int:
    """
    Determine the batch size for the dynamics evolution
    """
    can_be_batched = bindings.checkSuperOpBatchingCompatibility(
        hamiltonians) if isinstance(
            hamiltonians[0],
            SuperOperator) else bindings.checkBatchingCompatibility(
                hamiltonians, collapse_operators)
    input_type_str = "super-operators" if isinstance(
        hamiltonians[0],
        SuperOperator) else "Hamiltonian and collapse operators"

    if not can_be_batched:
        if max_batch_size is not None and max_batch_size > 1:
            raise ValueError(
                f"The input {input_type_str} are not compatible for batching. Unable to run batched simulation with the requested batch size {max_batch_size}."
            )
        if max_batch_size is None:
            print(
                f"Warning: The input {input_type_str} are not compatible for batching. Running the simulation in non-batched mode."
            )
        # Only run sequentially
        max_batch_size = 1

    # If the max_batch_size is not provided (or set to sequential mode above), use the number of input operators as the batch size.
    if max_batch_size is None:
        # Use the number of super-operators as the batch size.
        max_batch_size = len(hamiltonians)

    return max_batch_size


def split_simulation_batches(
        hamiltonians: Sequence[Operator] | Sequence[SuperOperator],
        collapse_operators: Sequence[Sequence[Operator]],
        initial_states: Sequence[cudaq_runtime.State], batch_size: int):
    """
    Split the simulation into batches based on the provided batch size.
    """
    # Split the Hamiltonians into batches
    ham_batches = [
        hamiltonians[i:i + batch_size]
        for i in range(0, len(hamiltonians), batch_size)
    ]
    if len(collapse_operators) == 0:
        # If no collapse operators are provided, use empty lists for batches.
        collapse_batches = [[]] * len(ham_batches)
    else:
        # Split the collapse operators into batches
        collapse_batches = [
            collapse_operators[i:i + batch_size]
            for i in range(0, len(collapse_operators), batch_size)
        ]
    initial_state_batches = [
        initial_states[i:i + batch_size]
        for i in range(0, len(initial_states), batch_size)
    ]

    return ham_batches, collapse_batches, initial_state_batches


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

    # Validate the arguments.
    hamiltonian, collapse_operators = validate_evolve_dynamics_args(
        hamiltonian, collapse_operators)

    # Determine if we need to use mixed states.
    has_collapse_operators = should_use_mixed_state(hamiltonian,
                                                    collapse_operators)

    is_super_op = isinstance(hamiltonian, SuperOperator) or (isinstance(
        hamiltonian, Sequence) and isinstance(hamiltonian[0], SuperOperator))
    if isinstance(hamiltonian, Sequence):
        # Batch Hamiltonian or super-operators
        batch_size_to_run = determine_batch_size(hamiltonian,
                                                 collapse_operators,
                                                 max_batch_size)
        if batch_size_to_run < len(hamiltonian):
            # Need to split the simulation into smaller batches.
            ham_batches, collapse_batches, initial_state_batches = split_simulation_batches(
                hamiltonian, collapse_operators, initial_state,
                batch_size_to_run)
            all_results = []
            for batch_idx in range(len(ham_batches)):
                # Run the simulation for each batch.
                batch_hamiltonian = ham_batches[batch_idx]
                batch_collapse_ops = collapse_batches[batch_idx]
                batch_initial_states = initial_state_batches[batch_idx]

                # Run the simulation for the current batch.
                result = evolve_dynamics(batch_hamiltonian, dimensions,
                                         schedule, batch_initial_states,
                                         batch_collapse_ops, observables,
                                         store_intermediate_results, integrator)
                if isinstance(result, Sequence):
                    # If the result is a sequence, append each result.
                    all_results.extend(result)
                else:
                    all_results.append(result)
            # Return the results for all batches.
            return all_results

    # Main simulation flow for single evolution or fully-batched evolution.
    if is_super_op:
        integrator.set_system(dimensions, schedule, hamiltonian)
    else:
        integrator.set_system(dimensions, schedule, hamiltonian,
                              collapse_operators)

    hilbert_space_dims_list = list(hilbert_space_dims)
    expectation_op = [
        bindings.CuDensityMatExpectation(MatrixOperator(observable),
                                         hilbert_space_dims_list)
        for observable in observables
    ]

    batch_size = 1
    is_batched_evolve = False
    if isinstance(initial_state, Sequence) and len(initial_state) > 1:
        batch_size = len(initial_state)
        initial_state = bindings.createBatchedState(initial_state,
                                                    hilbert_space_dims_list,
                                                    has_collapse_operators)
        is_batched_evolve = True
    else:
        initial_state = initial_state[0] if isinstance(
            initial_state, Sequence) else initial_state
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

    # Get MPI state for distributed mode handling
    is_mpi_init = cudaq_runtime.mpi.is_initialized()
    mpi_rank = cudaq_runtime.mpi.rank() if is_mpi_init else 0
    mpi_num_ranks = cudaq_runtime.mpi.num_ranks() if is_mpi_init else 1

    # We requires an even partition for distributed batched states.
    if batch_size > 1 and batch_size % mpi_num_ranks != 0:
        raise RuntimeError(
            f"Distributed batched states require an even partition across ranks: "
            f"batch size {batch_size} is not divisible by number of ranks {mpi_num_ranks}. Please adjust "
            "the number of MPI ranks or the batch size.")

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
                # In distributed mode, the split operation only returns the
                # local states held by this rank. The number of split states
                # may be less than batch_size.
                local_num_states = len(split_states)
                if local_num_states == batch_size:
                    # Non-distributed mode: all states are local
                    for i in range(batch_size):
                        intermediate_states[i].append(split_states[i])
                else:
                    # Distributed mode: only some states are local.
                    # Calculate the batch offset for this rank based on even
                    # distribution of states across ranks.
                    states_per_rank = batch_size // mpi_num_ranks
                    batch_offset = mpi_rank * states_per_rank
                    for i, split_state in enumerate(split_states):
                        global_idx = batch_offset + i
                        if global_idx < batch_size:
                            intermediate_states[global_idx].append(split_state)

    bindings.clearContext()

    # In distributed mode, only create results for states that have data.
    # Check if we're in distributed mode and filter accordingly.
    is_distributed = cudaq_runtime.mpi.is_initialized(
    ) and cudaq_runtime.mpi.num_ranks() > 1

    if is_distributed:
        # In distributed mode, each rank only has local states.
        # Return results only for the states this rank holds.
        local_results = []
        for i in range(batch_size):
            if intermediate_states[i]:  # Only include if we have data
                if store_intermediate_results != IntermediateResultSave.NONE:
                    local_results.append(
                        cudaq_runtime.EvolveResult(intermediate_states[i],
                                                   exp_vals[i]))
                else:
                    local_results.append(
                        cudaq_runtime.EvolveResult(intermediate_states[i][-1],
                                                   exp_vals[i][-1]))
        results = local_results
    else:
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
