# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import cudaq
import numpy as np


def _total_dimension(dimensions: Mapping[int, int]) -> int:
    """Return the product of all local Hilbert-space dimensions."""
    dimension = 1
    for local_dimension in dimensions.values():
        dimension *= local_dimension
    return dimension


def _identity_state(dimension: int):
    """Return |I> used to evolve closed-system propagators directly."""
    identity = np.eye(dimension, dtype=np.complex128).reshape(-1)
    return cudaq.State.from_data(identity)


def _basis_states(dimension: int):
    """Return Liouville basis states used to reconstruct Lindblad maps."""
    states = []
    for index in range(dimension):
        data = np.zeros(dimension, dtype=np.complex128)
        data[index] = 1.0
        states.append(cudaq.State.from_data(data))
    return states


def _state_to_matrix(state, dimension: int) -> np.ndarray:
    """Convert a vectorized propagated identity state back to a matrix."""
    data = np.array(state).reshape(-1)
    expected_size = dimension * dimension

    if data.size != expected_size:
        raise RuntimeError("Expected propagator state with size "
                           f"{expected_size}, got {data.size}.")

    return data.reshape((dimension, dimension)).T


def _closed_system_generator(hamiltonian):
    """Represent dU/dt = -i H U as left multiplication by -i H."""
    generator = cudaq.SuperOperator()
    generator += cudaq.SuperOperator.left_multiply(-1j * hamiltonian)
    return generator


def _extract_propagator(result, dimension: int,
                        store_intermediate_results: bool):
    """Extract closed-system propagators from a CUDA-Q evolve result."""
    if store_intermediate_results:
        return [
            _state_to_matrix(state, dimension)
            for state in result.intermediate_states()
        ]

    return _state_to_matrix(result.final_state(), dimension)


def _extract_batched_basis_propagator(results):
    """Stack evolved Liouville basis states into a dense Lindblad map."""
    columns = [
        np.array(single_result.final_state()).reshape(-1)
        for single_result in results
    ]
    return np.column_stack(columns)


def _is_operator_like(value) -> bool:
    """Return True for CUDA-Q operators accepted by the dynamics backend."""
    return hasattr(value, "to_matrix")


def _is_collapse_operator_batch(collapse_operators) -> bool:
    """Return True when collapse operators are grouped per Hamiltonian."""
    return bool(collapse_operators) and not _is_operator_like(
        collapse_operators[0])


def _collapse_operator_batches(collapse_operators, batch_size: int):
    """Broadcast collapse operators to match the Hamiltonian batch size."""
    if not collapse_operators:
        return [[] for _ in range(batch_size)]

    if _is_collapse_operator_batch(collapse_operators):
        if len(collapse_operators) != batch_size:
            raise ValueError("Batched collapse_operators must have the same "
                             "length as the Hamiltonian batch.")
        return [list(ops) for ops in collapse_operators]

    return [collapse_operators for _ in range(batch_size)]


def propagator(
    hamiltonian,
    dimensions: Mapping[int, int],
    schedule,
    *,
    collapse_operators=None,
    store_intermediate_results: bool = False,
    integrator=None,
    max_batch_size: Optional[int] = None,
):
    """Compute dynamics propagators.

    For closed-system dynamics, computes the matrix U satisfying the
    Schrodinger-picture propagator equation with initial condition U(t0) = I.

    For open-system dynamics with collapse operators, computes the Lindblad
    map S with initial condition S(t0) = I. This map acts on density
    matrices after matrix-to-vector reshaping and propagates rho(t0) to
    rho(t).

    Args:
        hamiltonian: CUDA-Q operator H(t), or a sequence of operators for
            batched propagator computation.
        dimensions: Mapping from degree-of-freedom index to local dimension.
        schedule: CUDA-Q dynamics schedule.
        collapse_operators: Optional sequence of Lindblad collapse operators.
            If provided, the helper returns the Lindblad map.
        store_intermediate_results: If True, return propagators at the
            intermediate schedule points saved by the dynamics backend.
        integrator: Optional dynamics integrator.
        max_batch_size: Optional maximum batch size for the dynamics backend.

    Returns:
        For closed-system dynamics, returns a dense complex NumPy array with
        shape ``(dim, dim)``.

        For open-system dynamics, returns a dense complex NumPy array with
        shape ``(dim**2, dim**2)``.

        If ``store_intermediate_results`` is True, returns a list of dense
        matrices. For a sequence of Hamiltonians, returns one such result per
        Hamiltonian.
    """
    collapse_operators = [] if collapse_operators is None else list(
        collapse_operators)

    is_batched = isinstance(hamiltonian,
                            Sequence) and not hasattr(hamiltonian, "to_matrix")
    hamiltonians = list(hamiltonian) if is_batched else [hamiltonian]
    collapse_operator_batches = _collapse_operator_batches(
        collapse_operators, len(hamiltonians))
    open_system = any(collapse_operator_batches)

    system_dimension = _total_dimension(dimensions)
    propagator_dimension = (system_dimension * system_dimension
                            if open_system else system_dimension)
    evolution_dimensions = dimensions

    if open_system:
        generators = hamiltonians
    else:
        generators = [_closed_system_generator(h) for h in hamiltonians]

    if open_system:
        initial_states = [
            _basis_states(propagator_dimension) for _ in generators
        ]
    else:
        initial_states = [
            _identity_state(propagator_dimension) for _ in generators
        ]

    save_mode = (cudaq.IntermediateResultSave.ALL if store_intermediate_results
                 else cudaq.IntermediateResultSave.NONE)

    evolve_collapse_operators = []
    if open_system:
        evolve_collapse_operators = (collapse_operator_batches if is_batched
                                     else collapse_operator_batches[0])

    evolve_generators = generators if is_batched else generators[0]
    evolve_initial_states = initial_states if is_batched else initial_states[0]

    if open_system and is_batched:
        evolve_generators = []
        evolve_initial_states = []
        evolve_collapse_operators = []
        for generator, basis_states, collapse_batch in zip(
                generators, initial_states, collapse_operator_batches):
            for basis_state in basis_states:
                evolve_generators.append(generator)
                evolve_initial_states.append(basis_state)
                evolve_collapse_operators.append(collapse_batch)

    result = cudaq.evolve(
        evolve_generators,
        evolution_dimensions,
        schedule,
        evolve_initial_states,
        collapse_operators=evolve_collapse_operators,
        observables=[],
        store_intermediate_results=save_mode,
        integrator=integrator,
        max_batch_size=max_batch_size,
    )

    if open_system:
        if is_batched:
            return [
                _extract_batched_basis_propagator(
                    result[index * propagator_dimension:(index + 1) *
                           propagator_dimension])
                for index in range(len(generators))
            ]
        return _extract_batched_basis_propagator(result)

    if is_batched:
        return [
            _extract_propagator(single_result, propagator_dimension,
                                store_intermediate_results)
            for single_result in result
        ]

    return _extract_propagator(result, propagator_dimension,
                               store_intermediate_results)
