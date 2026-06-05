from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np


def _total_dimension(dimensions: Mapping[int, int]) -> int:
    dimension = 1
    for local_dimension in dimensions.values():
        dimension *= local_dimension
    return dimension


def _identity_state(dimension: int):
    import cudaq

    identity = np.eye(dimension, dtype=np.complex128).reshape(-1)
    return cudaq.State.from_data(identity)


def _state_to_matrix(state, dimension: int) -> np.ndarray:
    data = np.array(state).reshape(-1)
    expected_size = dimension * dimension

    if data.size != expected_size:
        raise RuntimeError(
            "Expected propagator state with size "
            f"{expected_size}, got {data.size}.")

    return data.reshape((dimension, dimension))


def _closed_system_generator(hamiltonian):
    import cudaq

    generator = cudaq.SuperOperator()
    generator += cudaq.SuperOperator.left_multiply(-1j * hamiltonian)
    return generator


def _open_system_generator(hamiltonian, collapse_operators):
    import cudaq

    generator = cudaq.SuperOperator()
    generator += cudaq.SuperOperator.left_multiply(-1j * hamiltonian)
    generator += cudaq.SuperOperator.right_multiply(1j * hamiltonian)

    for collapse_operator in collapse_operators:
        collapse_operator_dagger = collapse_operator.dagger()
        collapse_operator_product = collapse_operator_dagger * collapse_operator

        generator += cudaq.SuperOperator.left_right_multiply(
            collapse_operator, collapse_operator_dagger)
        generator += cudaq.SuperOperator.left_multiply(
            -0.5 * collapse_operator_product)
        generator += cudaq.SuperOperator.right_multiply(
            -0.5 * collapse_operator_product)

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
    store_intermediate_results: bool = False,
    integrator=None,
    max_batch_size: Optional[int] = None,
):
    """Compute dynamics propagators.

    For closed-system dynamics, computes the matrix U satisfying

        dU/dt = -i H(t) U,  U(t_initial) = I.

    For open-system dynamics with collapse operators, computes the
    superoperator propagator S satisfying

        d vec(rho)/dt = L(t) vec(rho),  S(t_initial) = I,

    where L(t) is the Lindblad generator.
    """
    import cudaq

    collapse_operators = [] if collapse_operators is None else list(
        collapse_operators)
    open_system = len(collapse_operators) > 0

    system_dimension = _total_dimension(dimensions)
    propagator_dimension = (system_dimension * system_dimension
                            if open_system else system_dimension)

    is_batched = isinstance(hamiltonian, Sequence)
    hamiltonians = list(hamiltonian) if is_batched else [hamiltonian]

    if open_system:
        generators = [
            _open_system_generator(h, collapse_operators) for h in hamiltonians
        ]
    else:
        generators = [_closed_system_generator(h) for h in hamiltonians]

    initial_states = [_identity_state(propagator_dimension) for _ in generators]

    save_mode = (cudaq.IntermediateResultSave.ALL
                 if store_intermediate_results else
                 cudaq.IntermediateResultSave.NONE)

    result = cudaq.evolve(
        generators if is_batched else generators[0],
        dimensions,
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
