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


def _open_system_generator(hamiltonian, collapse_operators,
                           collapse_operator_adjoint_ops):
    generator = cudaq.SuperOperator()
    generator += cudaq.SuperOperator.left_multiply(-1j * hamiltonian)
    generator += cudaq.SuperOperator.right_multiply(1j * hamiltonian)

    for collapse_operator, collapse_operator_adjoint in zip(
            collapse_operators, collapse_operator_adjoint_ops):
        collapse_product = collapse_operator_adjoint * collapse_operator
        generator += cudaq.SuperOperator.left_right_multiply(
            collapse_operator, collapse_operator_adjoint)
        generator += cudaq.SuperOperator.left_multiply(-0.5 * collapse_product)
        generator += cudaq.SuperOperator.right_multiply(-0.5 * collapse_product)

    return generator


def _extract_propagator(result, dimension: int,
                        store_intermediate_results: bool):
    if store_intermediate_results:
        return [
            _state_to_matrix(state, dimension)
            for state in result.intermediate_states()
        ]

    return _state_to_matrix(result.final_state(), dimension)


def propagator(
    hamiltonian,
    dimensions: Mapping[int, int],
    schedule,
    *,
    collapse_operators=None,
    collapse_operator_adjoint_ops=None,
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
        collapse_operator_adjoint_ops: Optional sequence containing the adjoint
            operator for each collapse operator.
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
    collapse_operator_adjoint_ops = ([] if collapse_operator_adjoint_ops is None
                                     else list(collapse_operator_adjoint_ops))

    if collapse_operator_adjoint_ops and len(
            collapse_operator_adjoint_ops) != len(collapse_operators):
        raise ValueError("collapse_operator_adjoint_ops must have the same "
                         "length as collapse_operators.")

    open_system = len(collapse_operators) > 0
    if open_system and not collapse_operator_adjoint_ops:
        raise ValueError("collapse_operator_adjoint_ops must be provided for "
                         "open-system propagators.")

    system_dimension = _total_dimension(dimensions)
    propagator_dimension = (system_dimension * system_dimension
                            if open_system else system_dimension)
    evolution_dimensions = dimensions

    is_batched = isinstance(hamiltonian,
                            Sequence) and not hasattr(hamiltonian, "to_matrix")
    hamiltonians = list(hamiltonian) if is_batched else [hamiltonian]

    if open_system:
        generators = [
            _open_system_generator(h, collapse_operators,
                                   collapse_operator_adjoint_ops)
            for h in hamiltonians
        ]
    else:
        generators = [_closed_system_generator(h) for h in hamiltonians]

    initial_states = [_identity_state(propagator_dimension) for _ in generators]

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

    if is_batched:
        return [
            _extract_propagator(single_result, propagator_dimension,
                                store_intermediate_results)
            for single_result in result
        ]

    return _extract_propagator(result, propagator_dimension,
                               store_intermediate_results)
