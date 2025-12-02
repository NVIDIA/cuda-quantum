import cudaq
from cudaq import spin
from collections import Counter
import numpy as np
import cupy as cp


def get_basis_states_as_array(results, num_spins):

    ans = np.array([list(map(int, k)) for k in results.keys()])
    assert ans.shape == (len(results), num_spins)

    return ans


def calculate_cumulative_results(all_measurement_results):

    cumulative_results = []

    running = Counter()
    for d in all_measurement_results:
        running += Counter(d)  # add current dict to running total
        cumulative_results.append(dict(running))  # store a *copy* as plain dict

    return cumulative_results


def get_basis_states_and_index(sample_results):

    basis_states_and_index = {}
    for i, (k, v) in enumerate(sample_results.items()):
        binary_tuple = tuple(int(bit) for bit in k)  # convert '101' -> (1,0,1)
        basis_states_and_index[binary_tuple] = i  # assign new index

    return basis_states_and_index


def create_heisenberg_hamiltonian(n_spins: int, Jx: float, Jy: float, Jz: float,
                                  h_x: list[float], h_y: list[float],
                                  h_z: list[float]):

    ham = 0

    # Add two-qubit interaction terms for Heisenberg Hamiltonian
    for i in range(0, n_spins - 1):
        ham += Jx * spin.x(i) * spin.x(i + 1)
        ham += Jy * spin.y(i) * spin.y(i + 1)
        ham += Jz * spin.z(i) * spin.z(i + 1)

    return ham


def extract_coeffs_and_paulis(H: cudaq.SpinOperator):

    coefficients = []
    pauli_words = []
    n_qubits = H.qubit_count

    for term in H:

        coefficients.append(term.evaluate_coefficient())
        pauli_words.append(term.get_pauli_word(n_qubits))

    coefficients = np.array(coefficients)
    if np.allclose(coefficients.imag, 0):
        coefficients = coefficients.real

    return coefficients, pauli_words


def vectorized_projected_hamiltonian(basis_states, hamiltonian_pauli_words,
                                     hamiltonian_coefficients_numpy, use_gpu):
    """
    GPU-accelerated, `vectorized` implementation of projected_hamiltonian.
    
    Uses CuPy when GPU is available, otherwise falls back to NumPy.
    Produces bit-for-bit identical results to the original function.
    
    Args:
        basis_states: (n_basis, n_qubits) array of basis state bit vectors
        `hamiltonian_pauli_words: list of Pauli strings (e.g., ['XYZII', 'ZZXYI'])`
        `hamiltonian_coefficients_numpy: array of coefficients for each Pauli term`
        verbose: if True, print debug info (not used in accelerated version)
        `use_gpu: True (force GPU), or False (force CPU)`
    
    Returns:
        matrix_rows, matrix_cols, matrix_elements: lists matching original function
    """
    # Determine backend: CuPy (GPU) or NumPy (CPU)

    xp = cp if use_gpu else np

    n_basis = basis_states.shape[0]
    n_qubits = basis_states.shape[1]
    n_terms = len(hamiltonian_pauli_words)

    if n_basis == 0 or n_terms == 0:
        return [], [], []

    # Transfer data to device
    basis_states_xp = xp.asarray(basis_states, dtype=xp.int8)
    coefficients_xp = xp.asarray(hamiltonian_coefficients_numpy,
                                 dtype=xp.complex128)

    # Convert Pauli strings to numeric array: I=0, X=1, Y=2, Z=3
    pauli_to_int = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    pauli_ops_np = np.array([[pauli_to_int[p]
                              for p in pauli_word]
                             for pauli_word in hamiltonian_pauli_words],
                            dtype=np.int8)
    pauli_ops = xp.asarray(pauli_ops_np)  # (n_terms, n_qubits)

    # Broadcast for all (basis_state, `pauli_term`) combinations
    # states_expanded: (n_basis, 1, n_qubits)
    # `pauli_expanded`:  (1, n_terms, n_qubits)
    states_expanded = basis_states_xp[:, None, :]
    pauli_expanded = pauli_ops[None, :, :]

    # STEP 1: Compute transformed states (X and Y flip bits)
    flip_mask = (pauli_expanded == 1) | (pauli_expanded == 2)  # X=1 or Y=2
    transformed_states = xp.where(flip_mask, 1 - states_expanded,
                                  states_expanded)
    # transformed_states: (n_basis, n_terms, n_qubits)

    # STEP 2: Compute phase factors
    # Y on initial 0 → multiply by +1j
    # Y on initial 1 → multiply by -1j
    # Z on initial 1 → multiply by -1

    y_mask = (pauli_expanded == 2)
    z_mask = (pauli_expanded == 3)

    # Count phase contributions per (basis, term) pair
    n_y0 = xp.sum(y_mask & (states_expanded == 0), axis=2,
                  dtype=xp.int32)  # (n_basis, n_terms)
    n_y1 = xp.sum(y_mask & (states_expanded == 1), axis=2, dtype=xp.int32)
    n_z1 = xp.sum(z_mask & (states_expanded == 1), axis=2, dtype=xp.int32)

    # Compute phase index (mod 4) and look up phase value
    phase_index = (n_y0 - n_y1 + 2 * n_z1) % 4  # (n_basis, n_terms)
    phase_lookup = xp.array([1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j],
                            dtype=xp.complex128)
    phase_factors = phase_lookup[phase_index]  # (n_basis, n_terms)

    # STEP 3: Compute Hamiltonian matrix elements
    hamiltonian_elements = coefficients_xp[None, :] * phase_factors

    # STEP 4: Convert states to integers for fast lookup
    # Binary to integer: state_int = sum(bit[q] * 2^(n_qubits-1-q))
    powers_of_2 = xp.asarray(
        1 << np.arange(n_qubits - 1, -1, -1, dtype=np.int64))

    basis_ints = xp.sum(basis_states_xp.astype(xp.int64) * powers_of_2,
                        axis=1)  # (n_basis,)
    transformed_ints = xp.sum(transformed_states.astype(xp.int64) * powers_of_2,
                              axis=2)  # (n_basis, n_terms)

    # STEP 5: Find matching basis states using sorted search
    # Sort basis integers and use binary search for O(log n) lookup
    sorted_indices = xp.argsort(basis_ints)
    sorted_basis_ints = basis_ints[sorted_indices]

    transformed_flat = transformed_ints.ravel()  # (n_basis * n_terms,)

    # Binary search to find potential match positions
    search_positions = xp.searchsorted(sorted_basis_ints, transformed_flat)

    # Validate matches (`searchsorted` finds insertion point, not necessarily exact match)
    in_bounds = search_positions < n_basis
    search_positions_clipped = xp.minimum(search_positions, n_basis - 1)
    actually_found = in_bounds & (sorted_basis_ints[search_positions_clipped]
                                  == transformed_flat)

    # Map back to original indices
    final_indices = sorted_indices[search_positions_clipped]

    # STEP 6: Build output arrays with correct ordering
    # Column indices: for pair (i, j), col = i (the initial basis state index)
    col_indices = xp.repeat(xp.arange(n_basis, dtype=xp.int64), n_terms)

    # Filter to valid entries only
    matrix_rows_xp = final_indices[actually_found]
    matrix_cols_xp = col_indices[actually_found]
    matrix_elements_xp = hamiltonian_elements.ravel()[actually_found]

    # STEP 7: Return arrays in appropriate format
    if use_gpu:
        # Return CuPy arrays directly for GPU sparse matrix
        return matrix_rows_xp, matrix_cols_xp, matrix_elements_xp
    else:
        matrix_rows = matrix_rows_xp.tolist()
        matrix_cols = matrix_cols_xp.tolist()
        matrix_elements = matrix_elements_xp.tolist()
        return matrix_rows, matrix_cols, matrix_elements


def projected_hamiltonian(basis_states, hamiltonian_pauli_words,
                          hamiltonian_coefficients_numpy, verbose):

    matrix_rows, matrix_cols, matrix_elements = [], [], []

    for i in range(
            basis_states.shape[0]):  #loop over all basis states in subspace

        for pauli_operator, coefficient in zip(
                hamiltonian_pauli_words, hamiltonian_coefficients_numpy
        ):  #loop over each term in the Hamiltonian

            initial_state = basis_states[i]
            initial_state_idx = i

            transformed_state = initial_state.copy()
            phase_factor = 1.0 + 0j

            if verbose:
                print('initial_state            ', initial_state)
            if verbose:
                print('pauli_operator, coeff     ', pauli_operator, coefficient)

            for qubit_position, pauli_op in enumerate(
                    pauli_operator
            ):  #loop over each qubit in each term of the Hamiltonian
                # `print('pauli_op, qubit_position', pauli_op, qubit_position)`

                if pauli_op == 'I':
                    pass

                elif pauli_op == 'X':

                    transformed_state[
                        qubit_position] = 1 - transformed_state[qubit_position]

                elif pauli_op == 'Y':

                    if transformed_state[qubit_position] == 0:
                        phase_factor *= 1j
                    else:
                        phase_factor *= -1j

                    transformed_state[
                        qubit_position] = 1 - transformed_state[qubit_position]

                elif pauli_op == 'Z':
                    if transformed_state[qubit_position] == 1:
                        phase_factor *= -1

            if verbose:
                print('final_state, phase_factor', transformed_state,
                      phase_factor)
            if verbose:
                print(' ')

            hamiltonian_element = coefficient * phase_factor

            matches = np.all(basis_states == transformed_state, axis=1)
            final_index = np.where(matches)[0]

            if len(final_index) >= 1:
                matrix_rows.append(final_index[0])
                matrix_cols.append(initial_state_idx)
                matrix_elements.append(hamiltonian_element)

    return matrix_rows, matrix_cols, matrix_elements
