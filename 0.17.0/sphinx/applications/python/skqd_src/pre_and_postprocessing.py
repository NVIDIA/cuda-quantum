import time
import cudaq
from cudaq import spin
from collections import Counter
import numpy as np
import cupy as cp

from math import comb
from itertools import combinations

import matplotlib.pyplot as plt

# =====================================================================
#  Measurement helpers
# =====================================================================


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


# =====================================================================
#  Hamiltonian construction
# =====================================================================


def create_heisenberg_hamiltonian(n_spins: int, Jx: float, Jy: float,
                                  Jz: float):

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


# =====================================================================
#  Projected Hamiltonian — CSR (`vectorized` GPU/CPU)
# =====================================================================


def vectorized_projected_hamiltonian(basis_states, hamiltonian_pauli_words,
                                     hamiltonian_coefficients_numpy, use_gpu):
    """
    GPU-accelerated, `vectorized` implementation of projected_hamiltonian.
    
    Uses CuPy when GPU is available, otherwise falls back to NumPy.
    Produces bit-for-bit identical results to the original function.
    
    Args:
        basis_states: (n_basis, n_qubits) array of basis state bit vectors
        `hamiltonian_pauli_words`: list of Pauli strings (e.g., `['XYZII', 'ZZXYI']`)
        `hamiltonian_coefficients_numpy`: array of coefficients for each Pauli term
        `use_gpu`: True (force GPU), or False (force CPU)
    
    Returns:
        matrix_rows, matrix_cols, matrix_elements: lists matching original function
    """
    xp = cp if use_gpu else np

    n_basis = basis_states.shape[0]
    n_qubits = basis_states.shape[1]
    n_terms = len(hamiltonian_pauli_words)

    if n_basis == 0 or n_terms == 0:
        return [], [], []

    basis_states_xp = xp.asarray(basis_states, dtype=xp.int8)
    coefficients_xp = xp.asarray(hamiltonian_coefficients_numpy,
                                 dtype=xp.complex128)

    pauli_to_int = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    pauli_ops_np = np.array([[pauli_to_int[p]
                              for p in pauli_word]
                             for pauli_word in hamiltonian_pauli_words],
                            dtype=np.int8)
    pauli_ops = xp.asarray(pauli_ops_np)

    states_expanded = basis_states_xp[:, None, :]
    pauli_expanded = pauli_ops[None, :, :]

    flip_mask = (pauli_expanded == 1) | (pauli_expanded == 2)
    transformed_states = xp.where(flip_mask, 1 - states_expanded,
                                  states_expanded)

    y_mask = (pauli_expanded == 2)
    z_mask = (pauli_expanded == 3)

    n_y0 = xp.sum(y_mask & (states_expanded == 0), axis=2, dtype=xp.int32)
    n_y1 = xp.sum(y_mask & (states_expanded == 1), axis=2, dtype=xp.int32)
    n_z1 = xp.sum(z_mask & (states_expanded == 1), axis=2, dtype=xp.int32)

    del flip_mask, y_mask, z_mask

    phase_index = (n_y0 - n_y1 + 2 * n_z1) % 4
    phase_lookup = xp.array([1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j],
                            dtype=xp.complex128)
    phase_factors = phase_lookup[phase_index]

    hamiltonian_elements = coefficients_xp[None, :] * phase_factors

    powers_of_2 = xp.asarray(
        1 << np.arange(n_qubits - 1, -1, -1, dtype=np.int64))

    basis_ints = xp.sum(basis_states_xp.astype(xp.int64) * powers_of_2, axis=1)

    transformed_ints = xp.empty((n_basis, n_terms), dtype=xp.int64)
    for start in range(0, n_basis, 4096):
        transformed_ints[start:start +
                         4096] = (transformed_states[start:start + 4096].astype(
                             xp.int64) @ powers_of_2)
    del transformed_states

    sorted_indices = xp.argsort(basis_ints)
    sorted_basis_ints = basis_ints[sorted_indices]

    transformed_flat = transformed_ints.ravel()

    search_positions = xp.searchsorted(sorted_basis_ints, transformed_flat)

    in_bounds = search_positions < n_basis
    search_positions_clipped = xp.minimum(search_positions, n_basis - 1)
    actually_found = in_bounds & (sorted_basis_ints[search_positions_clipped]
                                  == transformed_flat)

    final_indices = sorted_indices[search_positions_clipped]

    col_indices = xp.repeat(xp.arange(n_basis, dtype=xp.int64), n_terms)

    matrix_rows_xp = final_indices[actually_found]
    matrix_cols_xp = col_indices[actually_found]
    matrix_elements_xp = hamiltonian_elements.ravel()[actually_found]

    if use_gpu:
        return matrix_rows_xp, matrix_cols_xp, matrix_elements_xp
    else:
        matrix_rows = matrix_rows_xp.tolist()
        matrix_cols = matrix_cols_xp.tolist()
        matrix_elements = matrix_elements_xp.tolist()
        return matrix_rows, matrix_cols, matrix_elements


# =====================================================================
#  Projected Hamiltonian — CSR (naive loop, CPU reference)
# =====================================================================


def projected_hamiltonian_cpu(basis_states, hamiltonian_pauli_words,
                              hamiltonian_coefficients_numpy):

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

            for qubit_position, pauli_op in enumerate(
                    pauli_operator
            ):  #loop over each qubit in each term of the Hamiltonian

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

            hamiltonian_element = coefficient * phase_factor

            matches = np.all(basis_states == transformed_state, axis=1)
            final_index = np.where(matches)[0]

            if len(final_index) >= 1:
                matrix_rows.append(final_index[0])
                matrix_cols.append(initial_state_idx)
                matrix_elements.append(hamiltonian_element)

    return matrix_rows, matrix_cols, matrix_elements


def _eigsh_solve_ritz(alpha, beta, beta_k, k, which):
    alpha = cp.asnumpy(alpha)
    beta = cp.asnumpy(beta)

    t = np.diag(alpha)
    t = t + np.diag(beta[:-1], k=1)
    t = t + np.diag(beta[:-1], k=-1)
    if beta_k is not None:
        beta_k = cp.asnumpy(beta_k)
        t[k, :k] = beta_k
        t[:k, k] = beta_k
    w, s = np.linalg.eigh(t)

    if which == 'LA':
        idx = np.argsort(w)
        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]
    elif which == 'LM':
        idx = np.argsort(np.absolute(w))
        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]
    elif which == 'SA':
        idx = np.argsort(w)
        wk = w[idx[:k]]
        sk = s[:, idx[:k]]

    return cp.array(wk), cp.array(sk)


# =====================================================================
#  Utility
# =====================================================================


def plot_skqd_convergence(csr_energies, lo_energies, dims,
                          exact_ground_state_energy):
    plt.figure(figsize=(5, 4))

    plt.plot(dims,
             csr_energies,
             'o-',
             linewidth=2,
             markersize=8,
             label='CSR eigsh')
    plt.plot(dims,
             lo_energies,
             's--',
             linewidth=2,
             markersize=8,
             label='Matrix-free Lanczos')
    plt.plot(dims, [exact_ground_state_energy] * len(dims),
             'g',
             linewidth=2,
             label='Exact ground state')

    plt.xticks(list(dims))
    plt.xlabel("Krylov Subspace Dimension", fontsize=12)
    plt.ylabel("Ground State Energy", fontsize=12)
    plt.title("SKQD Ground State Energy Convergence", fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    max_diff = max(abs(a - b) for a, b in zip(csr_energies, lo_energies))
    plt.text(
        0.02,
        0.98,
        f'Max solver difference: {max_diff:.2e}\nExact energy: {exact_ground_state_energy:.6f}',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.show()


def binary_strings_fixed_hamming_weight(n, k):
    """Generate all binary strings of length n with exactly k ones."""
    if k < 0 or k > n:
        return np.empty((0, n), dtype=bool)
    if k == 0:
        return np.zeros((1, n), dtype=bool)
    if k == n:
        return np.ones((1, n), dtype=bool)
    num_combinations = comb(n, k)
    result = np.zeros((num_combinations, n), dtype=bool)
    for idx, positions in enumerate(combinations(range(n), k)):
        result[idx, list(positions)] = True
    return result


# =====================================================================
#  GPU Hash Table for O(1) basis state lookup
# =====================================================================

_HASH_BUILD_KERNEL_CODE = r'''
extern "C" __global__
void build_hash_table(
    const long long* __restrict__ keys,
    long long* __restrict__ hash_keys,
    int* __restrict__ hash_vals,
    const int n_entries,
    const int hash_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_entries) return;

    long long key = keys[idx];
    unsigned int mask = (unsigned int)(hash_size - 1);
    unsigned int h = (unsigned int)(
        (unsigned long long)(key * 2654435761ULL) >> 32) & mask;

    for (int probe = 0; probe < 256; probe++) {
        unsigned long long old = atomicCAS(
            (unsigned long long*)&hash_keys[h],
            (unsigned long long)(0xFFFFFFFFFFFFFFFFULL),
            (unsigned long long)key);
        if (old == 0xFFFFFFFFFFFFFFFFULL || (long long)old == key) {
            hash_vals[h] = idx;
            return;
        }
        h = (h + 1) & mask;
    }
}
'''

_hash_build_kernel = cp.RawKernel(_HASH_BUILD_KERNEL_CODE, 'build_hash_table')


def build_gpu_hash_table(basis_ints_gpu, n_basis):
    """Build GPU-resident hash table: basis_int_value -> index.

    Uses open-addressing with linear probing and Knuth multiplicative hash.
    Load factor kept below 50% to avoid probe-chain overflows.
    """
    hash_size = 1
    while hash_size < 4 * n_basis:
        hash_size <<= 1

    hash_keys = cp.full(hash_size, -1, dtype=cp.int64)
    hash_vals = cp.full(hash_size, -1, dtype=cp.int32)

    threads = 256
    blocks = (n_basis + threads - 1) // threads
    _hash_build_kernel((blocks,), (threads,),
                       (basis_ints_gpu, hash_keys, hash_vals, np.int32(n_basis),
                        np.int32(hash_size)))
    cp.cuda.Device().synchronize()

    return hash_keys, hash_vals, hash_size


# =====================================================================
#  Row-Partitioned Tiled `Matvec` Kernel
#
#  Each thread processes TILE_SIZE Pauli terms for a single row.
#  Accumulates in registers and does a single `atomicAdd` per tile,
#  reducing contention by tile_size factor.
#
#  Row-partitioning means each GPU writes to disjoint result elements,
#  so no inter-GPU AllReduce is needed -- only AllGather.
# =====================================================================

_ROW_KERNEL_CODE = r'''
extern "C" __global__
void row_partitioned_matvec(
    const long long* __restrict__ basis_ints,
    const long long* __restrict__ hash_keys,
    const int*       __restrict__ hash_vals,
    const long long* __restrict__ flip_ints,
    const long long* __restrict__ y_mask_ints,
    const long long* __restrict__ z_mask_ints,
    const int*       __restrict__ n_y_total_arr,
    const double*    __restrict__ coeff_re,
    const double*    __restrict__ coeff_im,
    const double*    __restrict__ v_re,
    const double*    __restrict__ v_im,
    double*          __restrict__ result_re,
    double*          __restrict__ result_im,
    const int n_terms,
    const int my_row_start,
    const int my_n_rows,
    const int hash_size,
    const int tile_size)
{
    long long idx = (long long)blockIdx.y * gridDim.x * blockDim.x
                  + (long long)blockIdx.x * blockDim.x
                  + threadIdx.x;

    int n_tiles = (n_terms + tile_size - 1) / tile_size;
    long long total = (long long)my_n_rows * n_tiles;
    if (idx >= total) return;

    int local_row = (int)(idx % my_n_rows);
    int tile_idx  = (int)(idx / my_n_rows);
    int global_row = my_row_start + local_row;

    long long bi = basis_ints[global_row];
    unsigned int hash_mask = (unsigned int)(hash_size - 1);

    int term_start = tile_idx * tile_size;
    int term_end = term_start + tile_size;
    if (term_end > n_terms) term_end = n_terms;

    double acc_re = 0.0, acc_im = 0.0;

    for (int t = term_start; t < term_end; t++) {
        long long col_state = bi ^ flip_ints[t];

        unsigned int h = (unsigned int)(
            (unsigned long long)(col_state * 2654435761ULL) >> 32) & hash_mask;
        int col = -1;
        for (int probe = 0; probe < 256; probe++) {
            long long hk = hash_keys[h];
            if (hk == col_state) { col = hash_vals[h]; break; }
            if (hk == -1LL) break;
            h = (h + 1) & hash_mask;
        }
        if (col < 0) continue;

        int ny1 = __popcll(col_state & y_mask_ints[t]);
        int nz1 = __popcll(col_state & z_mask_ints[t]);
        int nyt  = n_y_total_arr[t];
        int pidx = ((nyt - 2 * ny1 + 2 * nz1) % 4 + 4) % 4;

        double pr, pi;
        switch (pidx) {
            case 0: pr =  1.0; pi =  0.0; break;
            case 1: pr =  0.0; pi =  1.0; break;
            case 2: pr = -1.0; pi =  0.0; break;
            default: pr = 0.0; pi = -1.0; break;
        }

        double vr = v_re[col], vi = v_im[col];
        double cr = coeff_re[t], ci = coeff_im[t];

        double cpr = cr * pr - ci * pi;
        double cpi = cr * pi + ci * pr;

        acc_re += cpr * vr - cpi * vi;
        acc_im += cpr * vi + cpi * vr;
    }

    if (acc_re != 0.0 || acc_im != 0.0) {
        atomicAdd(&result_re[local_row], acc_re);
        atomicAdd(&result_im[local_row], acc_im);
    }
}
'''

_row_kernel = cp.RawKernel(_ROW_KERNEL_CODE, 'row_partitioned_matvec')

# =====================================================================
#  Hamiltonian Data Preparation
# =====================================================================


def prepare_hamiltonian_data(basis_states, hamiltonian_pauli_words,
                             hamiltonian_coefficients_numpy):
    """Build all GPU-resident data structures for the Hamiltonian.

    Returns a dict of CuPy arrays needed by the row-partitioned kernel.
    """
    n_basis = basis_states.shape[0]
    n_qubits = basis_states.shape[1]
    n_terms = len(hamiltonian_pauli_words)

    basis_np = np.ascontiguousarray(basis_states, dtype=np.int8)
    powers_of_2 = cp.asarray(
        1 << np.arange(n_qubits - 1, -1, -1, dtype=np.int64))
    CHUNK = 2_000_000
    basis_ints = cp.empty(n_basis, dtype=cp.int64)
    for _start in range(0, n_basis, CHUNK):
        _end = min(_start + CHUNK, n_basis)
        _chunk = cp.asarray(basis_np[_start:_end], dtype=cp.int64)
        basis_ints[_start:_end] = cp.sum(_chunk * powers_of_2, axis=1)
        del _chunk
    cp.get_default_memory_pool().free_all_blocks()

    hash_keys, hash_vals, hash_size = build_gpu_hash_table(basis_ints, n_basis)

    pauli_to_int = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    pauli_ops_np = np.array(
        [[pauli_to_int[p] for p in pw] for pw in hamiltonian_pauli_words],
        dtype=np.int8)
    pauli_ops = cp.asarray(pauli_ops_np)

    flip_ints = cp.sum(
        ((pauli_ops == 1) | (pauli_ops == 2)).astype(cp.int64) * powers_of_2,
        axis=1)
    y_mask_ints = cp.sum((pauli_ops == 2).astype(cp.int64) * powers_of_2,
                         axis=1)
    z_mask_ints = cp.sum((pauli_ops == 3).astype(cp.int64) * powers_of_2,
                         axis=1)
    del pauli_ops, powers_of_2

    _popcount_and = cp.ElementwiseKernel('int64 a, int64 b', 'int32 y',
                                         'y = __popcll(a & b)', 'popcount_and')
    n_y_total = _popcount_and(y_mask_ints, y_mask_ints)

    coeffs_np = np.ascontiguousarray(hamiltonian_coefficients_numpy,
                                     dtype=np.complex128)
    coeff_re = cp.ascontiguousarray(cp.asarray(coeffs_np.real,
                                               dtype=cp.float64))
    coeff_im = cp.ascontiguousarray(cp.asarray(coeffs_np.imag,
                                               dtype=cp.float64))

    cp.get_default_memory_pool().free_all_blocks()

    return {
        'basis_ints': basis_ints,
        'hash_keys': hash_keys,
        'hash_vals': hash_vals,
        'hash_size': hash_size,
        'flip_ints': flip_ints,
        'y_mask_ints': y_mask_ints,
        'z_mask_ints': z_mask_ints,
        'n_y_total': n_y_total,
        'coeff_re': coeff_re,
        'coeff_im': coeff_im,
        'n_basis': n_basis,
        'n_terms': n_terms,
    }


# =====================================================================
#  Distributed `Eigsh` (Memory-Efficient Multi-GPU `Lanczos`)
#
#  `Lanczos` vectors are partitioned across GPUs so each GPU stores only
#  its row-partition chunk: V_local[ncv, chunk_size] instead of
#  V[ncv, n_basis].  This reduces per-GPU memory from O(ncv * n_basis)
#  to O(ncv * n_basis / `n_gpus`), enabling much larger problem sizes.
#
#  Scalar `Lanczos` quantities (alpha, beta) are kept identical on all
#  ranks via NCCL `AllReduce`, so every rank follows the same convergence
#  path and issues the same number of collective calls (deadlock-free).
# =====================================================================


def distributed_eigsh(ham_data,
                      rank,
                      size,
                      nccl_comm_obj,
                      k=1,
                      which='SA',
                      ncv=None,
                      maxiter=None,
                      tol=0,
                      v0_full=None,
                      tile_size=32,
                      return_eigenvectors=False):
    """Distributed thick-restart `Lanczos` with row-partitioned `matvec`.

    Returns
    -------
    eigenvalues : `cupy` array of shape (k,)
    (eigenvectors_local : `cupy` array, only if return_eigenvectors=True)
    """
    n_basis = ham_data['n_basis']
    n_terms = ham_data['n_terms']
    dtype = cp.complex128
    rdtype = cp.float64

    chunk_size = (n_basis + size - 1) // size
    my_start = rank * chunk_size
    my_end = min(my_start + chunk_size, n_basis)
    my_n_rows = my_end - my_start

    n = n_basis
    if ncv is None:
        ncv = min(max(2 * k, k + 32), n - 1)
    else:
        ncv = min(max(ncv, k + 2), n - 1)
    if maxiter is None:
        maxiter = 10 * n
    if tol == 0:
        tol = np.finfo(np.complex128).eps

    if size > 1:
        from cupy.cuda import nccl as _nccl

    stream_ptr = cp.cuda.get_current_stream().ptr

    V_local = cp.zeros((ncv, chunk_size), dtype=dtype)
    alpha = cp.zeros((ncv,), dtype=dtype)
    beta = cp.zeros((ncv,), dtype=rdtype)
    u_local = cp.zeros(chunk_size, dtype=dtype)
    v_cur = cp.zeros(chunk_size, dtype=dtype)

    allgather_len = size * chunk_size
    v_re_full = cp.empty(allgather_len, dtype=rdtype)
    v_im_full = cp.empty(allgather_len, dtype=rdtype)
    _result_re = cp.zeros(max(my_n_rows, 1), dtype=rdtype)
    _result_im = cp.zeros(max(my_n_rows, 1), dtype=rdtype)
    _send_re = cp.empty(chunk_size, dtype=rdtype)
    _send_im = cp.empty(chunk_size, dtype=rdtype)

    _scalar_buf = cp.empty(2, dtype=rdtype)
    _norm_buf = cp.empty(1, dtype=rdtype)

    n_tiles = (n_terms + tile_size - 1) // tile_size
    total_threads = my_n_rows * n_tiles if my_n_rows > 0 else 0
    tpb = 256
    if total_threads > 0:
        total_blocks = (total_threads + tpb - 1) // tpb
        max_gx = 2**31 - 1
        grid_y = max(1, (total_blocks + max_gx - 1) // max_gx)
        grid_x = min(total_blocks, max_gx)
    else:
        grid_x, grid_y = 1, 1

    def _matvec(v_loc):
        _send_re[:my_n_rows] = v_loc[:my_n_rows].real
        _send_im[:my_n_rows] = v_loc[:my_n_rows].imag
        if my_n_rows < chunk_size:
            _send_re[my_n_rows:] = 0
            _send_im[my_n_rows:] = 0

        if size > 1:
            nccl_comm_obj.allGather(_send_re.data.ptr, v_re_full.data.ptr,
                                    chunk_size, _nccl.NCCL_FLOAT64, stream_ptr)
            nccl_comm_obj.allGather(_send_im.data.ptr, v_im_full.data.ptr,
                                    chunk_size, _nccl.NCCL_FLOAT64, stream_ptr)
            cp.cuda.Device().synchronize()
        else:
            v_re_full[:n_basis] = v_loc[:n_basis].real
            v_im_full[:n_basis] = v_loc[:n_basis].imag

        _result_re[:my_n_rows] = 0
        _result_im[:my_n_rows] = 0

        if my_n_rows > 0 and n_terms > 0:
            _row_kernel((grid_x, grid_y), (tpb,),
                        (ham_data['basis_ints'], ham_data['hash_keys'],
                         ham_data['hash_vals'], ham_data['flip_ints'],
                         ham_data['y_mask_ints'], ham_data['z_mask_ints'],
                         ham_data['n_y_total'], ham_data['coeff_re'],
                         ham_data['coeff_im'], v_re_full, v_im_full, _result_re,
                         _result_im, np.int32(n_terms), np.int32(my_start),
                         np.int32(my_n_rows), np.int32(
                             ham_data['hash_size']), np.int32(tile_size)))
        cp.cuda.Device().synchronize()

        out = cp.zeros(chunk_size, dtype=dtype)
        out[:my_n_rows] = (_result_re[:my_n_rows] + 1j * _result_im[:my_n_rows])
        return out

    def _dotc(a_loc, b_loc):
        val = cp.vdot(a_loc[:my_n_rows], b_loc[:my_n_rows])
        if size > 1:
            _scalar_buf[0] = val.real
            _scalar_buf[1] = val.imag
            nccl_comm_obj.allReduce(_scalar_buf.data.ptr, _scalar_buf.data.ptr,
                                    2, _nccl.NCCL_FLOAT64, _nccl.NCCL_SUM,
                                    stream_ptr)
            cp.cuda.Device().synchronize()
            return _scalar_buf[0] + 1j * _scalar_buf[1]
        return val

    def _nrm2(a_loc):
        sq = cp.sum(a_loc[:my_n_rows].real**2 + a_loc[:my_n_rows].imag**2)
        if size > 1:
            _norm_buf[0] = sq
            nccl_comm_obj.allReduce(_norm_buf.data.ptr, _norm_buf.data.ptr, 1,
                                    _nccl.NCCL_FLOAT64, _nccl.NCCL_SUM,
                                    stream_ptr)
            cp.cuda.Device().synchronize()
            return cp.sqrt(_norm_buf[0])
        return cp.sqrt(sq)

    def _reorth(V_loc, u_loc, nv):
        uu_loc = V_loc[:nv].conj() @ u_loc
        if size > 1:
            uu_flat = cp.empty(nv * 2, dtype=rdtype)
            uu_flat[0::2] = uu_loc.real
            uu_flat[1::2] = uu_loc.imag
            nccl_comm_obj.allReduce(uu_flat.data.ptr, uu_flat.data.ptr, nv * 2,
                                    _nccl.NCCL_FLOAT64, _nccl.NCCL_SUM,
                                    stream_ptr)
            cp.cuda.Device().synchronize()
            uu = uu_flat[0::2] + 1j * uu_flat[1::2]
        else:
            uu = uu_loc
        u_loc -= V_loc[:nv].T @ uu
        return uu

    def _allreduce_complex(vec, nv):
        if size > 1:
            buf = cp.empty(nv * 2, dtype=rdtype)
            buf[0::2] = vec.real
            buf[1::2] = vec.imag
            nccl_comm_obj.allReduce(buf.data.ptr, buf.data.ptr, nv * 2,
                                    _nccl.NCCL_FLOAT64, _nccl.NCCL_SUM,
                                    stream_ptr)
            cp.cuda.Device().synchronize()
            return buf[0::2] + 1j * buf[1::2]
        return vec

    if v0_full is not None:
        v_cur[:my_n_rows] = v0_full[my_start:my_end]
    else:
        cp.random.seed(42)
        v_cur[:my_n_rows] = cp.random.random(my_n_rows,
                                             dtype=rdtype).astype(dtype)
    nrm = _nrm2(v_cur)
    V_local[0] = v_cur / nrm

    def _lanczos(i_start, i_end):
        nonlocal u_local
        v_cur[...] = V_local[i_start]
        for i in range(i_start, i_end):
            u_local[...] = _matvec(v_cur)
            alpha[i] = _dotc(v_cur, u_local)
            u_local -= alpha[i] * v_cur
            u_local -= beta[i - 1] * V_local[i - 1]
            uu = _reorth(V_local, u_local, i + 1)
            alpha[i] = alpha[i] + uu[i]
            beta[i] = _nrm2(u_local)
            if i < i_end - 1:
                v_cur[...] = u_local / beta[i]
                V_local[i + 1] = v_cur

    _lanczos(0, ncv)

    iter_count = ncv
    w, s = _eigsh_solve_ritz(alpha, beta, None, k, which)
    x_local = V_local.T @ s
    beta_k = beta[-1] * s[-1, :]
    res = float(cp.asnumpy(cp.linalg.norm(beta_k)))

    n_restarts = 0

    while res > tol and iter_count < maxiter:
        beta[:k] = 0
        alpha[:k] = w
        V_local[:k] = x_local.T

        uu_k = _allreduce_complex(V_local[:k].conj() @ u_local, k)
        u_local -= V_local[:k].T @ uu_k

        nrm = _nrm2(u_local)
        V_local[k] = u_local / nrm

        u_local[...] = _matvec(V_local[k])
        alpha[k] = _dotc(V_local[k], u_local)
        u_local -= alpha[k] * V_local[k]
        u_local -= V_local[:k].T @ beta_k
        beta[k] = _nrm2(u_local)
        V_local[k + 1] = u_local / beta[k]

        _lanczos(k + 1, ncv)
        iter_count += ncv - k
        w, s = _eigsh_solve_ritz(alpha, beta, beta_k, k, which)
        x_local = V_local.T @ s

        beta_k = beta[-1] * s[-1, :]
        res = float(cp.asnumpy(cp.linalg.norm(beta_k)))
        n_restarts += 1

    if return_eigenvectors:
        idx = cp.argsort(w)
        return w[idx], x_local[:, idx]
    else:
        return cp.sort(w),
