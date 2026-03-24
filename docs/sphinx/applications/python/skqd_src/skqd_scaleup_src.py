"""
Optimized Multi-GPU SKQD Eigenvalue Solver with Strong Scaling

Techniques inspired by Walkup et al. (arXiv:2601.16169, 2026):
  1. Row-partitioned matvec: each GPU computes H@v for disjoint rows
     -> perfect load balance, no AllReduce needed
  2. Symmetric multi-GPU Lanczos: all ranks run eigsh identically
     -> eliminates worker idle time
  3. NCCL AllGather (replaces Broadcast+AllReduce): ~3x less communication
  4. GPU hash table: O(1) basis state lookup vs O(log n) binary search
  5. Tiled kernel: each thread accumulates TILE_SIZE terms,
     reducing atomicAdd contention by TILE_SIZE factor

Usage:
    # Single GPU:
    python skqd_scaleup.py --molecule mol.json --n_configs 200000

    # Multi-GPU (MPI):
    mpirun -np 8 --bind-to none --allow-run-as-root python skqd_scaleup.py \\
        --molecule mol.json --n_configs 200000
"""

import argparse
import time
import numpy as np
import cupy as cp
from math import comb
from itertools import combinations
import json
import os
import resource
from datetime import datetime

try:
    import nvtx
    _NVTX_AVAILABLE = True
except ImportError:
    _NVTX_AVAILABLE = False

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh

from scipy.sparse import csr_matrix as scipy_csr_matrix
from scipy.sparse.linalg import eigsh as scipy_eigsh

# Try to import MPI; fall back to single-GPU mode if unavailable
try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except ImportError:
    _MPI_AVAILABLE = False


class _NvtxNoOp:
    """Drop-in replacement when nvtx is not installed."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

def _nvtx_range(name, color="blue"):
    if _NVTX_AVAILABLE:
        return nvtx.annotate(name, color=color)
    return _NvtxNoOp()


# ---- GPU-to-CPU transfer tracker (updated by _eigsh_solve_ritz) ----
_transfer_stats = {
    'ritz_calls': 0,
    'ritz_time': 0.0,
    'gpu_to_cpu_bytes': 0,
    'gpu_to_cpu_count': 0,
    'cpu_to_gpu_bytes': 0,
    'cpu_to_gpu_count': 0,
}


def _eigsh_solve_ritz(alpha, beta, beta_k, k, which):
    t_ritz_start = time.perf_counter()
    _transfer_stats['ritz_calls'] += 1

    # GPU -> CPU transfers for alpha and beta
    alpha_bytes = alpha.nbytes
    beta_bytes = beta.nbytes
    alpha = cp.asnumpy(alpha)
    beta = cp.asnumpy(beta)
    _transfer_stats['gpu_to_cpu_bytes'] += alpha_bytes + beta_bytes
    _transfer_stats['gpu_to_cpu_count'] += 2

    t = np.diag(alpha)
    t = t + np.diag(beta[:-1], k=1)
    t = t + np.diag(beta[:-1], k=-1)
    if beta_k is not None:
        bk_bytes = beta_k.nbytes
        beta_k = cp.asnumpy(beta_k)
        _transfer_stats['gpu_to_cpu_bytes'] += bk_bytes
        _transfer_stats['gpu_to_cpu_count'] += 1
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

    # CPU -> GPU transfers for results
    wk_gpu = cp.array(wk)
    sk_gpu = cp.array(sk)
    _transfer_stats['cpu_to_gpu_bytes'] += wk.nbytes + sk.nbytes
    _transfer_stats['cpu_to_gpu_count'] += 2

    t_ritz_end = time.perf_counter()
    _transfer_stats['ritz_time'] += t_ritz_end - t_ritz_start

    return wk_gpu, sk_gpu


# =====================================================================
#  Utility
# =====================================================================

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
#  Projected Hamiltonian (naive loop, CPU)
# =====================================================================

def projected_hamiltonian(basis_states, hamiltonian_pauli_words,
                          hamiltonian_coefficients_numpy, verbose=False):
    """Build projected Hamiltonian as scipy CSR matrix using naive loops.

    Slow reference implementation -- useful for correctness checks on
    small subspaces.
    """
    matrix_rows, matrix_cols, matrix_elements = [], [], []
    subspace_dimension = basis_states.shape[0]

    for i in range(subspace_dimension):
        for pauli_operator, coefficient in zip(
                hamiltonian_pauli_words, hamiltonian_coefficients_numpy):

            initial_state = basis_states[i]
            transformed_state = initial_state.copy()
            phase_factor = 1.0 + 0j

            if verbose:
                print('initial_state            ', initial_state)
                print('pauli_operator, coeff     ', pauli_operator, coefficient)

            for qubit_position, pauli_op in enumerate(pauli_operator):
                if pauli_op == 'I':
                    pass
                elif pauli_op == 'X':
                    transformed_state[qubit_position] = (
                        1 - transformed_state[qubit_position])
                elif pauli_op == 'Y':
                    if transformed_state[qubit_position] == 0:
                        phase_factor *= 1j
                    else:
                        phase_factor *= -1j
                    transformed_state[qubit_position] = (
                        1 - transformed_state[qubit_position])
                elif pauli_op == 'Z':
                    if transformed_state[qubit_position] == 1:
                        phase_factor *= -1

            if verbose:
                print('final_state, phase_factor', transformed_state,
                      phase_factor)
                print(' ')

            hamiltonian_element = coefficient * phase_factor

            matches = np.all(basis_states == transformed_state, axis=1)
            final_index = np.where(matches)[0]

            if len(final_index) >= 1:
                matrix_rows.append(final_index[0])
                matrix_cols.append(i)
                matrix_elements.append(hamiltonian_element)

    return scipy_csr_matrix(
        (matrix_elements, (matrix_rows, matrix_cols)),
        shape=(subspace_dimension, subspace_dimension))


# =====================================================================
#  Vectorized Projected Hamiltonian (GPU or CPU)
# =====================================================================

def vectorized_projected_hamiltonian(basis_states, hamiltonian_pauli_words,
                                     hamiltonian_coefficients_numpy,
                                     use_gpu=True):
    """GPU-accelerated vectorized construction of the projected Hamiltonian.

    Returns a CSR matrix (cupy if use_gpu=True, scipy otherwise).
    Produces bit-for-bit identical results to projected_hamiltonian.
    """
    xp = cp if use_gpu else np

    n_basis = basis_states.shape[0]
    n_qubits = basis_states.shape[1]
    n_terms = len(hamiltonian_pauli_words)

    if n_basis == 0 or n_terms == 0:
        csr_fn = cupy_csr_matrix if use_gpu else scipy_csr_matrix
        return csr_fn((n_basis, n_basis), dtype=np.complex128)

    if use_gpu:
        cp.get_default_memory_pool().free_all_blocks()

    basis_states = np.ascontiguousarray(basis_states, dtype=np.int8)
    hamiltonian_coefficients_numpy = np.ascontiguousarray(
        hamiltonian_coefficients_numpy, dtype=np.complex128)

    basis_states_xp = xp.asarray(basis_states, dtype=xp.int8)
    coefficients_xp = xp.asarray(hamiltonian_coefficients_numpy,
                                 dtype=xp.complex128)

    pauli_to_int = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    pauli_ops_np = np.array(
        [[pauli_to_int[p] for p in pw] for pw in hamiltonian_pauli_words],
        dtype=np.int8)
    pauli_ops = xp.asarray(pauli_ops_np)

    powers_of_2 = xp.asarray(
        1 << np.arange(n_qubits - 1, -1, -1, dtype=np.int64))

    basis_ints = xp.sum(
        basis_states_xp.astype(xp.int64) * powers_of_2, axis=1)
    del basis_states_xp

    flip_ints = xp.sum(
        ((pauli_ops == 1) | (pauli_ops == 2)).astype(xp.int64) * powers_of_2,
        axis=1)
    y_mask_ints = xp.sum(
        (pauli_ops == 2).astype(xp.int64) * powers_of_2, axis=1)
    z_mask_ints = xp.sum(
        (pauli_ops == 3).astype(xp.int64) * powers_of_2, axis=1)
    del pauli_ops

    transformed_ints = basis_ints[:, None] ^ flip_ints[None, :]
    del flip_ints

    if use_gpu:
        _popcount_and = cp.ElementwiseKernel(
            'int64 a, int64 b', 'int32 y',
            'y = __popcll(a & b)', 'popcount_and')
        n_y_total = _popcount_and(y_mask_ints, y_mask_ints)
        n_y1 = _popcount_and(basis_ints[:, None], y_mask_ints[None, :])
        n_z1 = _popcount_and(basis_ints[:, None], z_mask_ints[None, :])
    else:
        def _numpy_popcount(x):
            x = x.astype(np.uint64)
            x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
            x = ((x & np.uint64(0x3333333333333333))
                 + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333)))
            x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
            return ((x * np.uint64(0x0101010101010101))
                    >> np.uint64(56)).astype(np.int32)

        n_y_total = _numpy_popcount(y_mask_ints)
        n_y1 = _numpy_popcount(basis_ints[:, None] & y_mask_ints[None, :])
        n_z1 = _numpy_popcount(basis_ints[:, None] & z_mask_ints[None, :])

    del y_mask_ints, z_mask_ints

    phase_index = (n_y_total[None, :] - 2 * n_y1 + 2 * n_z1) % 4
    phase_lookup = xp.array([1.0+0j, 0.0+1j, -1.0+0j, 0.0-1j],
                            dtype=xp.complex128)
    phase_factors = phase_lookup[phase_index]
    del n_y1, n_z1, n_y_total, phase_index

    hamiltonian_elements = coefficients_xp[None, :] * phase_factors
    del phase_factors

    sorted_indices = xp.argsort(basis_ints)
    sorted_basis_ints = basis_ints[sorted_indices]

    transformed_flat = transformed_ints.ravel()

    search_positions = xp.searchsorted(sorted_basis_ints, transformed_flat)
    search_positions_clipped = xp.minimum(search_positions, n_basis - 1)
    in_bounds = search_positions < n_basis
    actually_found = in_bounds & (
        sorted_basis_ints[search_positions_clipped] == transformed_flat)

    final_indices = sorted_indices[search_positions_clipped]

    col_indices = xp.repeat(xp.arange(n_basis, dtype=xp.int64), n_terms)

    matrix_rows_xp = final_indices[actually_found]
    matrix_cols_xp = col_indices[actually_found]
    matrix_elements_xp = hamiltonian_elements.ravel()[actually_found]

    if use_gpu:
        H_csr = cupy_csr_matrix(
            (matrix_elements_xp, (matrix_rows_xp, matrix_cols_xp)),
            shape=(n_basis, n_basis))
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    else:
        H_csr = scipy_csr_matrix(
            (matrix_elements_xp.tolist(),
             (matrix_rows_xp.tolist(), matrix_cols_xp.tolist())),
            shape=(n_basis, n_basis))

    return H_csr


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
    Load factor ~70% ensures ~2.2 probes on average while halving memory.
    """
    hash_size = 1
    while hash_size < int(1.5 * n_basis):
        hash_size <<= 1

    # Initialize with sentinel value -1 (0xFFFF...F)
    hash_keys = cp.full(hash_size, -1, dtype=cp.int64)
    hash_vals = cp.full(hash_size, -1, dtype=cp.int32)

    threads = 256
    blocks = (n_basis + threads - 1) // threads
    _hash_build_kernel(
        (blocks,), (threads,),
        (basis_ints_gpu, hash_keys, hash_vals,
         np.int32(n_basis), np.int32(hash_size))
    )
    cp.cuda.Device().synchronize()

    return hash_keys, hash_vals, hash_size


# =====================================================================
#  Row-Partitioned Tiled Matvec Kernel
#
#  Each thread processes TILE_SIZE Pauli terms for a single row.
#  Accumulates in registers and does a single atomicAdd per tile,
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

    /* Map: rows as inner dim for coalesced basis_ints access
       and reduced atomicAdd contention (consecutive threads
       write to different result elements) */
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
        /* Column state = row state XOR flip (symmetric under XOR) */
        long long col_state = bi ^ flip_ints[t];

        /* Hash table lookup for column index: O(1) avg */
        unsigned int h = (unsigned int)(
            (unsigned long long)(col_state * 2654435761ULL) >> 32) & hash_mask;
        int col = -1;
        for (int probe = 0; probe < 256; probe++) {
            long long hk = hash_keys[h];
            if (hk == col_state) { col = hash_vals[h]; break; }
            if (hk == -1LL) break;  /* empty slot -> not found */
            h = (h + 1) & hash_mask;
        }
        if (col < 0) continue;

        /* Phase factor (depends on column/initial state per Slater rules).
           col_state = basis_ints[col] since col is the looked-up index. */
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

        /* Read input vector at column position */
        double vr = v_re[col], vi = v_im[col];
        double cr = coeff_re[t], ci = coeff_im[t];

        /* Complex multiply: element = coeff * phase * v[col] */
        double cpr = cr * pr - ci * pi;
        double cpi = cr * pi + ci * pr;

        acc_re += cpr * vr - cpi * vi;
        acc_im += cpr * vi + cpi * vr;
    }

    /* Single atomicAdd per tile (instead of per term) */
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

    # Basis states -> integer bitmasks (chunked to cap intermediate memory)
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

    # Build hash table for O(1) lookup
    t_hash_start = time.perf_counter()
    hash_keys, hash_vals, hash_size = build_gpu_hash_table(basis_ints, n_basis)
    t_hash_end = time.perf_counter()

    # Pauli strings -> bitmasks
    pauli_to_int = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    pauli_ops_np = np.array(
        [[pauli_to_int[p] for p in pw] for pw in hamiltonian_pauli_words],
        dtype=np.int8)
    pauli_ops = cp.asarray(pauli_ops_np)

    flip_ints = cp.sum(
        ((pauli_ops == 1) | (pauli_ops == 2)).astype(cp.int64) * powers_of_2,
        axis=1)
    y_mask_ints = cp.sum(
        (pauli_ops == 2).astype(cp.int64) * powers_of_2, axis=1)
    z_mask_ints = cp.sum(
        (pauli_ops == 3).astype(cp.int64) * powers_of_2, axis=1)
    del pauli_ops, powers_of_2

    _popcount_and = cp.ElementwiseKernel(
        'int64 a, int64 b', 'int32 y',
        'y = __popcll(a & b)', 'popcount_and')
    n_y_total = _popcount_and(y_mask_ints, y_mask_ints)

    coeffs_np = np.ascontiguousarray(
        hamiltonian_coefficients_numpy, dtype=np.complex128)
    coeff_re = cp.ascontiguousarray(
        cp.asarray(coeffs_np.real, dtype=cp.float64))
    coeff_im = cp.ascontiguousarray(
        cp.asarray(coeffs_np.imag, dtype=cp.float64))

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
        'hash_build_time': t_hash_end - t_hash_start,
    }


# =====================================================================
#  Distributed Eigsh (Memory-Efficient Multi-GPU Lanczos)
#
#  Lanczos vectors are partitioned across GPUs so each GPU stores only
#  its row-partition chunk: V_local[ncv, chunk_size] instead of
#  V[ncv, n_basis].  This reduces per-GPU memory from O(ncv * n_basis)
#  to O(ncv * n_basis / n_gpus), enabling much larger problem sizes.
#
#  Scalar Lanczos quantities (alpha, beta) are kept identical on all
#  ranks via NCCL AllReduce, so every rank follows the same convergence
#  path and issues the same number of collective calls (deadlock-free).
# =====================================================================

def distributed_eigsh(ham_data, rank, size, nccl_comm_obj,
                      k=1, which='SA', ncv=None, maxiter=None, tol=0,
                      v0_full=None, tile_size=32, return_eigenvectors=False):
    """Distributed thick-restart Lanczos with row-partitioned matvec.

    Returns
    -------
    eigenvalues : cupy array of shape (k,)
    timing      : dict with matvec timing breakdown
    part_info   : dict with partition metadata
    (eigenvectors_local : cupy array, only if return_eigenvectors=True)
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

    # ---- Distributed Lanczos vectors (chunk_size per GPU) ----
    V_local = cp.zeros((ncv, chunk_size), dtype=dtype)
    alpha = cp.zeros((ncv,), dtype=dtype)
    beta = cp.zeros((ncv,), dtype=rdtype)
    u_local = cp.zeros(chunk_size, dtype=dtype)
    v_cur = cp.zeros(chunk_size, dtype=dtype)

    # ---- Matvec buffers ----
    allgather_len = size * chunk_size
    v_re_full = cp.empty(allgather_len, dtype=rdtype)
    v_im_full = cp.empty(allgather_len, dtype=rdtype)
    _result_re = cp.zeros(max(my_n_rows, 1), dtype=rdtype)
    _result_im = cp.zeros(max(my_n_rows, 1), dtype=rdtype)
    _send_re = cp.empty(chunk_size, dtype=rdtype)
    _send_im = cp.empty(chunk_size, dtype=rdtype)

    # ---- AllReduce scratch ----
    _scalar_buf = cp.empty(2, dtype=rdtype)
    _norm_buf = cp.empty(1, dtype=rdtype)

    # ---- Kernel grid ----
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

    timing = {'n_calls': 0, 'compute_total': 0.0,
              'comm_total': 0.0, 'total': 0.0,
              'matvec_times': [],
              'allgather_total': 0.0, 'allgather_count': 0,
              'allreduce_total': 0.0, 'allreduce_count': 0,
              'reorth_total': 0.0, 'reorth_count': 0,
              'kernel_event_times_ms': []}

    # CUDA events for precise kernel timing
    _ev_start = cp.cuda.Event()
    _ev_end = cp.cuda.Event()

    # ---- Distributed matvec: AllGather v, kernel, return local ----
    def _matvec(v_loc):
        with _nvtx_range("matvec", color="green"):
            timing['n_calls'] += 1
            t0 = time.perf_counter()

            _send_re[:my_n_rows] = v_loc[:my_n_rows].real
            _send_im[:my_n_rows] = v_loc[:my_n_rows].imag
            if my_n_rows < chunk_size:
                _send_re[my_n_rows:] = 0
                _send_im[my_n_rows:] = 0

            t_r0 = time.perf_counter()
            with _nvtx_range("allgather", color="yellow"):
                if size > 1:
                    nccl_comm_obj.allGather(
                        _send_re.data.ptr, v_re_full.data.ptr,
                        chunk_size, _nccl.NCCL_FLOAT64, stream_ptr)
                    nccl_comm_obj.allGather(
                        _send_im.data.ptr, v_im_full.data.ptr,
                        chunk_size, _nccl.NCCL_FLOAT64, stream_ptr)
                    cp.cuda.Device().synchronize()
                else:
                    v_re_full[:n_basis] = v_loc[:n_basis].real
                    v_im_full[:n_basis] = v_loc[:n_basis].imag
            t_r1 = time.perf_counter()
            dt_ag = t_r1 - t_r0
            timing['comm_total'] += dt_ag
            timing['allgather_total'] += dt_ag
            timing['allgather_count'] += 1

            _result_re[:my_n_rows] = 0
            _result_im[:my_n_rows] = 0

            t_c0 = time.perf_counter()
            with _nvtx_range("kernel", color="red"):
                _ev_start.record()
                if my_n_rows > 0 and n_terms > 0:
                    _row_kernel(
                        (grid_x, grid_y), (tpb,),
                        (ham_data['basis_ints'], ham_data['hash_keys'],
                         ham_data['hash_vals'],
                         ham_data['flip_ints'], ham_data['y_mask_ints'],
                         ham_data['z_mask_ints'],
                         ham_data['n_y_total'], ham_data['coeff_re'],
                         ham_data['coeff_im'],
                         v_re_full, v_im_full,
                         _result_re, _result_im,
                         np.int32(n_terms), np.int32(my_start),
                         np.int32(my_n_rows),
                         np.int32(ham_data['hash_size']),
                         np.int32(tile_size)))
                _ev_end.record()
                _ev_end.synchronize()
                timing['kernel_event_times_ms'].append(
                    cp.cuda.get_elapsed_time(_ev_start, _ev_end))
            cp.cuda.Device().synchronize()
            t_c1 = time.perf_counter()
            timing['compute_total'] += t_c1 - t_c0

            out = cp.zeros(chunk_size, dtype=dtype)
            out[:my_n_rows] = (
                _result_re[:my_n_rows] + 1j * _result_im[:my_n_rows])
            t1 = time.perf_counter()
            dt = t1 - t0
            timing['total'] += dt
            timing['matvec_times'].append(dt)
            return out

    # ---- Distributed dot product: conj(a) . b ----
    def _dotc(a_loc, b_loc):
        val = cp.vdot(a_loc[:my_n_rows], b_loc[:my_n_rows])
        if size > 1:
            _scalar_buf[0] = val.real
            _scalar_buf[1] = val.imag
            t_ar0 = time.perf_counter()
            with _nvtx_range("allreduce_dotc", color="orange"):
                nccl_comm_obj.allReduce(
                    _scalar_buf.data.ptr, _scalar_buf.data.ptr,
                    2, _nccl.NCCL_FLOAT64, _nccl.NCCL_SUM, stream_ptr)
                cp.cuda.Device().synchronize()
            dt_ar = time.perf_counter() - t_ar0
            timing['allreduce_total'] += dt_ar
            timing['allreduce_count'] += 1
            timing['comm_total'] += dt_ar
            return _scalar_buf[0] + 1j * _scalar_buf[1]
        return val

    # ---- Distributed norm ----
    def _nrm2(a_loc):
        sq = cp.sum(a_loc[:my_n_rows].real ** 2
                    + a_loc[:my_n_rows].imag ** 2)
        if size > 1:
            _norm_buf[0] = sq
            t_ar0 = time.perf_counter()
            with _nvtx_range("allreduce_nrm2", color="orange"):
                nccl_comm_obj.allReduce(
                    _norm_buf.data.ptr, _norm_buf.data.ptr,
                    1, _nccl.NCCL_FLOAT64, _nccl.NCCL_SUM, stream_ptr)
                cp.cuda.Device().synchronize()
            dt_ar = time.perf_counter() - t_ar0
            timing['allreduce_total'] += dt_ar
            timing['allreduce_count'] += 1
            timing['comm_total'] += dt_ar
            return cp.sqrt(_norm_buf[0])
        return cp.sqrt(sq)

    # ---- Reorthogonalize u against V[:nv], return correction ----
    def _reorth(V_loc, u_loc, nv):
        t_ro0 = time.perf_counter()
        with _nvtx_range("reorth", color="purple"):
            uu_loc = V_loc[:nv].conj() @ u_loc
            if size > 1:
                uu_flat = cp.empty(nv * 2, dtype=rdtype)
                uu_flat[0::2] = uu_loc.real
                uu_flat[1::2] = uu_loc.imag
                t_ar0 = time.perf_counter()
                with _nvtx_range("allreduce_reorth", color="orange"):
                    nccl_comm_obj.allReduce(
                        uu_flat.data.ptr, uu_flat.data.ptr,
                        nv * 2, _nccl.NCCL_FLOAT64, _nccl.NCCL_SUM,
                        stream_ptr)
                    cp.cuda.Device().synchronize()
                dt_ar = time.perf_counter() - t_ar0
                timing['allreduce_total'] += dt_ar
                timing['allreduce_count'] += 1
                timing['comm_total'] += dt_ar
                uu = uu_flat[0::2] + 1j * uu_flat[1::2]
            else:
                uu = uu_loc
            u_loc -= V_loc[:nv].T @ uu
        dt_ro = time.perf_counter() - t_ro0
        timing['reorth_total'] += dt_ro
        timing['reorth_count'] += 1
        return uu

    # ---- Distributed AllReduce for small complex vectors ----
    def _allreduce_complex(vec, nv):
        if size > 1:
            buf = cp.empty(nv * 2, dtype=rdtype)
            buf[0::2] = vec.real
            buf[1::2] = vec.imag
            t_ar0 = time.perf_counter()
            with _nvtx_range("allreduce_complex", color="orange"):
                nccl_comm_obj.allReduce(
                    buf.data.ptr, buf.data.ptr,
                    nv * 2, _nccl.NCCL_FLOAT64, _nccl.NCCL_SUM,
                    stream_ptr)
                cp.cuda.Device().synchronize()
            dt_ar = time.perf_counter() - t_ar0
            timing['allreduce_total'] += dt_ar
            timing['allreduce_count'] += 1
            timing['comm_total'] += dt_ar
            return buf[0::2] + 1j * buf[1::2]
        return vec

    # ---- Initialize v0 ----
    if v0_full is not None:
        v_cur[:my_n_rows] = v0_full[my_start:my_end]
    else:
        cp.random.seed(42)
        v_cur[:my_n_rows] = cp.random.random(
            my_n_rows, dtype=rdtype).astype(dtype)
    nrm = _nrm2(v_cur)
    V_local[0] = v_cur / nrm

    # ---- Lanczos inner loop (used for initial pass and restarts) ----
    def _lanczos(i_start, i_end):
        nonlocal u_local
        v_cur[...] = V_local[i_start]
        for i in range(i_start, i_end):
            with _nvtx_range(f"lanczos_iter_{i}", color="cyan"):
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

    # ---- Initial Lanczos pass ----
    with _nvtx_range("lanczos_initial", color="blue"):
        _lanczos(0, ncv)

    iter_count = ncv
    w, s = _eigsh_solve_ritz(alpha, beta, None, k, which)
    x_local = V_local.T @ s
    beta_k = beta[-1] * s[-1, :]
    res = float(cp.asnumpy(cp.linalg.norm(beta_k)))

    convergence_residuals = [res]
    n_restarts = 0

    # ---- Thick-restart loop ----
    while res > tol and iter_count < maxiter:
        with _nvtx_range(f"restart_{n_restarts}", color="blue"):
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
            convergence_residuals.append(res)
            n_restarts += 1

    part_info = {
        'chunk_size': chunk_size,
        'my_start': my_start,
        'my_end': my_end,
        'my_n_rows': my_n_rows,
        'n_tiles': n_tiles,
        'tile_size': tile_size,
    }

    timing['convergence_residuals'] = convergence_residuals
    timing['n_restarts'] = n_restarts
    timing['final_residual'] = res
    timing['iter_count'] = iter_count

    if return_eigenvectors:
        idx = cp.argsort(w)
        return w[idx], x_local[:, idx], timing, part_info
    else:
        return cp.sort(w), timing, part_info



