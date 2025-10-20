import cudaq
from cudaq import spin

from collections import Counter
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def transform_state_with_pauli_operator(computational_state,
                                        pauli_operator_string):
    """
    Transform a computational basis state by applying a Pauli operator string.
    
    This function computes how a Pauli string acts on a computational basis state.
    For example, applying "`XY`" to |01⟩ gives i|10⟩ (bit flip + phase).
    
    Implementation follows MSB convention: `pauli_operator_string[i]` acts on computational_state[i]
    where i=0 corresponds to the most significant bit (leftmost position).
    
    Args:
        computational_state: Binary array representing |0⟩ and |1⟩ states in MSB order
        `pauli_operator_string`: String of Pauli operators (I, X, Y, Z) in MSB order
        
    Returns:
        tuple: (transformed_state, quantum_phase) where quantum_phase is ±1 or ±1j
    """
    transformed_state = computational_state.copy()
    quantum_phase = 1.0 + 0j  # Start with no phase; Y operators will add ±i phases

    # Apply each Pauli operator to its corresponding qubit
    for qubit_position, pauli_op in enumerate(pauli_operator_string):
        if pauli_op == 'I':
            # Identity: no change to state or phase
            continue
        elif pauli_op == 'X':
            # Pauli-X: bit flip |0⟩ ↔ |1⟩
            transformed_state[
                qubit_position] = 1 - transformed_state[qubit_position]
        elif pauli_op == 'Y':
            # Pauli-Y = i·X·Z: bit flip with phase ±i depending on initial state
            if transformed_state[qubit_position] == 0:
                quantum_phase *= 1j  # Y|0⟩ = i|1⟩
            else:
                quantum_phase *= -1j  # Y|1⟩ = -i|0⟩
            transformed_state[
                qubit_position] = 1 - transformed_state[qubit_position]
        elif pauli_op == 'Z':
            # Pauli-Z: phase flip for |1⟩ state
            if transformed_state[qubit_position] == 1:
                quantum_phase *= -1  # Z|1⟩ = -|1⟩, Z|0⟩ = |0⟩

    return transformed_state, quantum_phase


def construct_hamiltonian_in_subspace(pauli_operator_list,
                                      hamiltonian_coefficients,
                                      subspace_basis_states):
    """
    Project Hamiltonian operator onto the computational subspace spanned by basis states.
    
    This is the heart of SKQD: instead of computing expensive quantum matrix elements,
    we compute how each Pauli term in the Hamiltonian acts on our sampled basis states.
    Only matrix elements between states in our subspace contribute to the final result.
    
    Native CUDA-Q implementation using MSB (Most Significant Bit first) convention:
    - Both Pauli operator strings and basis states follow MSB bit ordering
    - Pauli operator index 0 acts on qubit 0 (leftmost bit position)
    
    Args:
        `pauli_operator_list: List of Pauli operator strings (e.g., ['IIXY', 'ZIII'])`
        hamiltonian_coefficients: List of real coefficients for each Pauli operator
        subspace_basis_states: Array of computational basis states defining the subspace
        
    Returns:
        `scipy.sparse matrix representing projected Hamiltonian within the subspace`
    """
    subspace_dimension = subspace_basis_states.shape[0]

    # Create fast lookup: basis state → subspace index
    # This allows O(1) checking if a transformed state is in our subspace
    state_to_index_map = {}
    for idx, basis_state in enumerate(subspace_basis_states):
        state_key = tuple(basis_state)
        state_to_index_map[state_key] = idx

    # Build sparse matrix in COO format (rows, cols, values)
    matrix_rows, matrix_cols, matrix_elements = [], [], []

    # For each basis state |i⟩ in our subspace...
    for initial_state_idx, initial_state in enumerate(subspace_basis_states):
        # For each Pauli term P_k in the Hamiltonian...
        for pauli_operator, coefficient in zip(pauli_operator_list,
                                               hamiltonian_coefficients):
            # Compute P_k|i⟩ = phase × |j⟩
            final_state, phase_factor = transform_state_with_pauli_operator(
                initial_state, pauli_operator)
            final_state_key = tuple(final_state)

            # Only keep matrix element if |j⟩ is also in our subspace
            if final_state_key in state_to_index_map:
                final_state_idx = state_to_index_map[final_state_key]
                matrix_rows.append(final_state_idx)
                matrix_cols.append(initial_state_idx)

                # Matrix element: ⟨j|H|i⟩ = coefficient × phase_factor
                hamiltonian_element = coefficient * phase_factor

                # Clean up tiny imaginary parts (should be real for Hermitian H)
                if abs(hamiltonian_element.imag) < 1e-14:
                    hamiltonian_element = hamiltonian_element.real
                matrix_elements.append(hamiltonian_element)

    # Convert to efficient sparse matrix format
    return csr_matrix((matrix_elements, (matrix_rows, matrix_cols)),
                      shape=(subspace_dimension, subspace_dimension))


def diagonalize_subspace_hamiltonian(subspace_basis_states,
                                     pauli_operator_list,
                                     hamiltonian_coefficients,
                                     verbose=False,
                                     **solver_options):
    """
    Perform eigenvalue decomposition of Hamiltonian within the computational subspace.
    
    This function combines Hamiltonian projection and `diagonalization` into one step.
    It's the final piece of SKQD: finding the lowest eigenvalues in our Krylov subspace.
    
    Args:
        subspace_basis_states: Array of computational basis states defining the subspace
        `pauli_operator_list: List of Pauli operator strings (e.g., ['IIXY', 'ZIII'])`
        hamiltonian_coefficients: List of real coefficients for each Pauli operator
        verbose: Enable diagnostic output
        `**solver_options: Additional arguments for scipy.sparse.linalg.eigsh`
        
    Returns:
        `numpy` array of eigenvalues from the subspace `diagonalization`
    """
    if subspace_basis_states.shape[0] == 0:
        return np.array([])

    # Step 1: Project the full Hamiltonian onto our sampled subspace
    projected_hamiltonian = construct_hamiltonian_in_subspace(
        pauli_operator_list, hamiltonian_coefficients, subspace_basis_states)

    if verbose:
        print(f"Subspace dimension: {projected_hamiltonian.shape[0]}")
        print(
            f"Hamiltonian sparsity: {projected_hamiltonian.nnz / (projected_hamiltonian.shape[0]**2):.4f}"
        )

    # Step 2: Find eigenvalues of the projected Hamiltonian
    try:
        # Use sparse eigensolver for efficiency (typically we only need a few eigenvalues)
        eigenvalues = eigsh(projected_hamiltonian,
                            return_eigenvectors=False,
                            **solver_options)
        return eigenvalues
    except Exception as solver_error:
        if verbose:
            print(f"Sparse eigensolver failed: {solver_error}")

        # Fallback: use dense `diagonalization` for small matrices
        if projected_hamiltonian.shape[0] <= 100:
            dense_hamiltonian = projected_hamiltonian.toarray()
            eigenvalues = np.linalg.eigvals(dense_hamiltonian)

            # Extract only the requested eigenvalues to match sparse solver behavior
            num_eigenvalues = solver_options.get('k', min(6, len(eigenvalues)))
            eigenvalue_selection = solver_options.get('which', 'SA')
            if eigenvalue_selection == 'SA':  # smallest algebraic eigenvalues
                selected_indices = np.argsort(eigenvalues)[:num_eigenvalues]
            else:
                selected_indices = np.argsort(eigenvalues)[-num_eigenvalues:]
            return eigenvalues[selected_indices]
        else:
            raise


def accumulate_krylov_measurements(measurement_results_sequence,
                                   krylov_dimension):
    """
    Progressively accumulate measurement outcomes from Krylov state evolution.
    
    This is a key insight of SKQD: instead of treating each Krylov state separately,
    we combine measurements from |ψ⟩, U|ψ⟩, U²|ψ⟩, ... to build increasingly
    rich computational `subspaces` that better approximate the true Krylov space.
    
    Args:
        measurement_results_sequence: List of measurement dictionaries from each U^k|ψ⟩
        krylov_dimension: Number of Krylov states to consider
        
    Returns:
        List of accumulated measurement dictionaries, where entry k contains
        measurements from all states |ψ⟩, U|ψ⟩, ..., U^k|ψ⟩
    """
    accumulated_measurements = []

    # For each Krylov dimension k = 1, 2, 3, ...
    for evolution_step in range(krylov_dimension):
        measurement_accumulator = Counter()

        # Combine measurements from all states up to U^k|ψ⟩
        for measurement_data in measurement_results_sequence[:evolution_step +
                                                             1]:
            measurement_accumulator.update(measurement_data)

        # Convert back to dictionary and store
        combined_measurements = dict(measurement_accumulator)
        accumulated_measurements.append(combined_measurements)

    return accumulated_measurements


def construct_xyz_spin_hamiltonian(
        system_size: int,
        interaction_strengths: tuple[float, float, float] = (1.0, 1.0, 1.0),
        external_field_strengths: tuple[float, float, float] = (0.0, 0.0, 0.0),
        topology_type: str = "ring") -> cudaq.SpinOperator:
    """
    Construct `XYZ` spin model Hamiltonian using native CUDA-Q SpinOperator framework.
    
    Implements the quantum many-body Hamiltonian:
    H = sum_{(i,j) ∈ edges} [J_x σ_i^x σ_j^x + J_y σ_i^y σ_j^y + J_z σ_i^z σ_j^z] +
        sum_{i ∈ sites} [h_x σ_i^x + h_y σ_i^y + h_z σ_i^z]
    
    Args:
        system_size: Number of quantum spins in the system
        interaction_strengths: (J_x, J_y, J_z) nearest-neighbor coupling parameters
        external_field_strengths: (h_x, h_y, h_z) local magnetic field components
        topology_type: Lattice connectivity ("ring" implements periodic boundary conditions)
        
    Returns:
        CUDA-Q SpinOperator encoding the XYZ spin Hamiltonian
    """
    J_x, J_y, J_z = interaction_strengths
    h_x, h_y, h_z = external_field_strengths

    # Initialize Hamiltonian with null operator
    spin_hamiltonian = 0.0 * spin.z(0)

    # Construct nearest-neighbor interaction terms (ring topology only)
    for site_i in range(system_size):
        site_j = (site_i +
                  1) % system_size  # Nearest neighbor with periodic wrapping

        # Add Pauli tensor product interactions (skip zero coefficients)
        if J_x != 0.0:
            spin_hamiltonian += J_x * spin.x(site_i) * spin.x(site_j)
        if J_y != 0.0:
            spin_hamiltonian += J_y * spin.y(site_i) * spin.y(site_j)
        if J_z != 0.0:
            spin_hamiltonian += J_z * spin.z(site_i) * spin.z(site_j)

    # Add local magnetic field terms (skip zero fields)
    if h_x != 0.0 or h_y != 0.0 or h_z != 0.0:
        for site_i in range(system_size):
            if h_x != 0.0:
                spin_hamiltonian += h_x * spin.x(site_i)
            if h_y != 0.0:
                spin_hamiltonian += h_y * spin.y(site_i)
            if h_z != 0.0:
                spin_hamiltonian += h_z * spin.z(site_i)

    return spin_hamiltonian


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


def extract_hamiltonian_data(spin_operator: cudaq.SpinOperator):
    """Extract coefficients, Pauli words, and strings from CUDA-Q SpinOperator.
    
    Optimized single-pass extraction of all required Hamiltonian data.
    
    Args:
        spin_operator: CUDA-Q SpinOperator to decompose
        
    Returns:
        `tuple: (coefficients_list, pauli_words_list, pauli_strings_list)`
    """
    system_size = spin_operator.qubit_count
    coefficients_list = []
    pauli_words_list = []
    pauli_strings_list = []

    for pauli_term in spin_operator:
        # Extract coefficient
        term_coefficient = pauli_term.evaluate_coefficient()
        assert abs(
            term_coefficient.imag
        ) < 1e-10, f"Non-real coefficient encountered: {term_coefficient}"
        coefficients_list.append(float(term_coefficient.real))

        # Extract Pauli string
        pauli_string = pauli_term.get_pauli_word(system_size)
        pauli_strings_list.append(pauli_string)

        # Create Pauli word object
        pauli_words_list.append(cudaq.pauli_word(pauli_string))

    return coefficients_list, pauli_words_list, pauli_strings_list


def create_tfim_hamiltonian(n_spins: int, h_field: float):
    """Create the Hamiltonian operator"""
    ham = 0

    # Add single-qubit terms
    for i in range(0, n_spins):
        ham += -1 * h_field * spin.x(i)

    # Add two-qubit interaction terms for Ising Hamiltonian
    for i in range(0, n_spins - 1):
        ham += -1 * spin.z(i) * spin.z(i + 1)

    return ham


def extract_basis_states_from_measurements(measurement_counts):
    """
    Extract computational basis states from CUDA-Q measurement results.
    
    This function converts the measurement outcome dictionary into a matrix where
    each row represents a unique computational basis state observed during sampling.
    
    Args:
        measurement_counts: Dictionary mapping bitstring outcomes to their frequencies
        
    Returns:
        `numpy` array of computational basis states (MSB ordering)
        `Shape: (num_unique_states, num_qubits)`
    """
    if not measurement_counts:
        return np.array([])

    # Extract all unique bitstrings that were observed during measurements
    observed_bitstrings = list(measurement_counts.keys())
    num_qubits = len(observed_bitstrings[0])

    # Convert bitstrings to binary matrix representation
    # Each row is a computational basis state |00...⟩, |01...⟩, etc.
    basis_state_matrix = np.array(
        [[int(bit) for bit in bitstring] for bitstring in observed_bitstrings],
        dtype=np.int8)

    return basis_state_matrix
