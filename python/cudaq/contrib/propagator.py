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
    dimension = 1
    for local_dimension in dimensions.values():
        dimension *= local_dimension
    return dimension


def _identity_state(dimension: int):
    identity = np.eye(dimension, dtype=np.complex128).reshape(-1)
    return cudaq.State.from_data(identity)


def _basis_states(dimension: int):
    states = []
    for index in range(dimension):
        data = np.zeros(dimension, dtype=np.complex128)
        data[index] = 1.0
        states.append(cudaq.State.from_data(data))
    return states


def _state_to_matrix(state, dimension: int) -> np.ndarray:
    data = np.array(state).reshape(-1)
    expected_size = dimension * dimension

    if data.size != expected_size:
        raise RuntimeError("Expected propagator state with size "
                           f"{expected_size}, got {data.size}.")

    return data.reshape((dimension, dimension)).T


def _closed_system_generator(hamiltonian):
    generator = cudaq.SuperOperator()
    generator += cudaq.SuperOperator.left_multiply(-1j * hamiltonian)
    return generator


def _extract_propagator(result, dimension: int,
                        store_intermediate_results: bool):
    if store_intermediate_results:
        return [
            _state_to_matrix(state, dimension)
            for state in result.intermediate_states()
        ]

    return _state_to_matrix(result.final_state(), dimension)


def _extract_batched_basis_propagator(results):
    columns = [
        np.array(single_result.final_state()).reshape(-1)
        for single_result in results
    ]
    return np.column_stack(columns)


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
    open_system = len(collapse_operators) > 0

    system_dimension = _total_dimension(dimensions)
    propagator_dimension = (system_dimension * system_dimension
                            if open_system else system_dimension)
    evolution_dimensions = dimensions

    is_batched = isinstance(hamiltonian,
                            Sequence) and not hasattr(hamiltonian, "to_matrix")
    hamiltonians = list(hamiltonian) if is_batched else [hamiltonian]

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

    result = cudaq.evolve(
        generators if is_batched else generators[0],
        evolution_dimensions,
        schedule,
        initial_states if is_batched else initial_states[0],
        collapse_operators=[],
        observables=[],
        store_intermediate_results=save_mode,
        integrator=integrator,
        max_batch_size=max_batch_size,
    )

    if open_system:
        if is_batched:
            return [
                _extract_batched_basis_propagator(single_result)
                for single_result in result
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
