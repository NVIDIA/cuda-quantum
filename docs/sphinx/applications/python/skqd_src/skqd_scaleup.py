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
    
from skqd_scaleup_src import *

from skqd_scaleup_src import _nvtx_range, _MPI_AVAILABLE, _NVML_AVAILABLE, _NVTX_AVAILABLE, _transfer_stats, _row_kernel

# from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
# from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh

# from scipy.sparse import csr_matrix as scipy_csr_matrix
# from scipy.sparse.linalg import eigsh as scipy_eigsh

# Try to import MPI; fall back to single-GPU mode if unavailable
try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except ImportError:
    _MPI_AVAILABLE = False

# ---- MPI setup ----
if _MPI_AVAILABLE:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    comm = None
    rank = 0
    size = 1

# ---- GPU assignment: one GPU per rank ----
n_devices = cp.cuda.runtime.getDeviceCount()
my_gpu = rank % n_devices
cp.cuda.Device(my_gpu).use()

t_wall_start = time.perf_counter()

# ---- Parse arguments ----
parser = argparse.ArgumentParser(
    description='Optimized Multi-GPU SKQD Eigenvalue Solver')
parser.add_argument('--molecule', default='SiH4_6-31G_14.json',
                    help='Path to molecule JSON file')
parser.add_argument('--n_configs', type=int, default=200000,
                    help='Number of basis configurations')
parser.add_argument('--tile_size', type=int, default=32,
                    help='Number of Pauli terms per kernel tile')
parser.add_argument('--output_csv', default='results.csv',
                    help='Path to output CSV file (appended)')
args = parser.parse_args()

# ---- Load molecule data (all ranks) ----
t_load_start = time.perf_counter()
with open(args.molecule, 'r') as f:
    data = json.load(f)
    nao = data['nao']
    nelec = data['nelec']
    fci_energy = data.get('fci_energy')
    pauli_words = list(data['hamiltonian'].keys())
    hamiltonian_coefficients_numpy = np.array(
        list(data['hamiltonian'].values()))
t_load_end = time.perf_counter()

# ---- Build basis states (all ranks, identical) ----
t_basis_start = time.perf_counter()

if 'TS_ROHF_AVAS_26' in os.path.basename(args.molecule):

    n_configs = args.n_configs
    configs_data = np.load('ts_102m_configs.npz')
    basis_states = configs_data['configs'][:n_configs]

else:

    alph_configs = binary_strings_fixed_hamming_weight(nao, nelec[0])
    beta_configs = binary_strings_fixed_hamming_weight(nao, nelec[1])

    n_configs = min(args.n_configs,
                    len(alph_configs) * len(beta_configs))
    basis_states = np.zeros((n_configs, 2 * nao), dtype=bool)

    count = 0
    for i in range(len(alph_configs)):
        for j in range(len(beta_configs)):
            if count >= n_configs:
                break
            basis_states[count, 0::2] = alph_configs[i]
            basis_states[count, 1::2] = beta_configs[j]
            count += 1
        if count >= n_configs:
            break


subspace_dimension = basis_states.shape[0]
num_qubits = basis_states.shape[1]
n_terms = len(pauli_words)
t_basis_end = time.perf_counter()

# ---- Print header (rank 0 only) ----
if rank == 0:
    print("=" * 60)
    print("  Row-Partitioned Multi-GPU SKQD Benchmark")
    print("=" * 60)
    print(f"  Molecule:       {args.molecule}")
    if fci_energy is not None:
        print(f"  FCI energy:     {fci_energy:.10f}")
    else:
        print(f"  FCI energy:     N/A")
    print(f"  n_configs:      {n_configs}")
    print(f"  num_qubits:     {num_qubits}")
    print(f"  n_pauli_terms:  {n_terms}")
    print(f"  GPUs:           {size}")
    print(f"  Tile size:      {args.tile_size}")
    print(f"  Hash table:     ON")
    print(f"  Row partition:  ON (distributed Lanczos)")

if comm is not None:
    comm.Barrier()

# ---- Prepare Hamiltonian data on GPU ----
with _nvtx_range("hamiltonian_setup", color="blue"):
    t_setup_start = time.perf_counter()
    ham_data = prepare_hamiltonian_data(
        basis_states, pauli_words, hamiltonian_coefficients_numpy)
    cp.cuda.Device(my_gpu).synchronize()
    t_setup_data = time.perf_counter()

# GPU memory after Hamiltonian setup
mempool = cp.get_default_memory_pool()
gpu_mem_after_setup_bytes = mempool.used_bytes()
gpu_mem_after_setup_mb = gpu_mem_after_setup_bytes / (1024 * 1024)

# ---- NCCL communicator ----
nccl_comm_obj = None
if size > 1:
    from cupy.cuda import nccl
    if rank == 0:
        nccl_id = nccl.get_unique_id()
    else:
        nccl_id = None
    nccl_id = comm.bcast(nccl_id, root=0)
    nccl_comm_obj = nccl.NcclCommunicator(size, nccl_id, rank)
t_setup_nccl = time.perf_counter()

if comm is not None:
    comm.Barrier()
t_setup_end = time.perf_counter()

# ---- GPU hardware state before eigsh (NVML) ----
nvml_before = {}
_nvml_handle = None
if _NVML_AVAILABLE:
    try:
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(my_gpu)
        nvml_before['sm_clock_mhz'] = pynvml.nvmlDeviceGetClockInfo(
            _nvml_handle, pynvml.NVML_CLOCK_SM)
        nvml_before['mem_clock_mhz'] = pynvml.nvmlDeviceGetClockInfo(
            _nvml_handle, pynvml.NVML_CLOCK_MEM)
        nvml_before['temperature_c'] = pynvml.nvmlDeviceGetTemperature(
            _nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
        nvml_before['power_mw'] = pynvml.nvmlDeviceGetPowerUsage(
            _nvml_handle)
        nvml_before['gpu_name'] = pynvml.nvmlDeviceGetName(_nvml_handle)
        if isinstance(nvml_before['gpu_name'], bytes):
            nvml_before['gpu_name'] = nvml_before['gpu_name'].decode()
    except Exception:
        _nvml_handle = None

# ---- Kernel occupancy and register pressure ----
kernel_attrs = {}
try:
    _kern = _row_kernel
    _kattrs = _kern.attributes
    kernel_attrs['num_regs'] = _kattrs.get('num_regs', -1)
    kernel_attrs['shared_size_bytes'] = _kattrs.get(
        'shared_size_bytes', -1)
    kernel_attrs['max_threads_per_block'] = _kattrs.get(
        'max_threads_per_block', -1)
except Exception:
    pass

if rank == 0:
    load_factor = n_configs / ham_data['hash_size'] * 100
    print(f"  Hash table size: {ham_data['hash_size']} "
            f"(load={load_factor:.1f}%)")
    _chunk = (subspace_dimension + size - 1) // size
    _last_rows = subspace_dimension - (size - 1) * _chunk
    print(f"  Rows/GPU:       ~{_chunk} "
            f"(last GPU: {_last_rows})")
    print(f"  GPU mem setup:  {gpu_mem_after_setup_mb:.1f} MB")
    print("-" * 60)
    print(f"  [TIMING] Data load:      "
            f"{t_load_end - t_load_start:.3f} s")
    print(f"  [TIMING] Basis build:    "
            f"{t_basis_end - t_basis_start:.3f} s")
    print(f"  [TIMING] GPU data setup: "
            f"{t_setup_data - t_setup_start:.3f} s")
    if size > 1:
        print(f"  [TIMING] NCCL init:      "
                f"{t_setup_nccl - t_setup_data:.3f} s")
    print(f"  [TIMING] Total setup:    "
            f"{t_setup_end - t_setup_start:.3f} s")
    print("-" * 60)

# ---- Deterministic starting vector (identical on all ranks) ----
np.random.seed(42)
v0_np = np.random.random((subspace_dimension,)).astype(np.float64)
v0_np = v0_np / np.linalg.norm(v0_np)
v0 = cp.asarray(v0_np, dtype=cp.complex128)

# ---- Reset transfer stats before eigsh ----
_transfer_stats['ritz_calls'] = 0
_transfer_stats['ritz_time'] = 0.0
_transfer_stats['gpu_to_cpu_bytes'] = 0
_transfer_stats['gpu_to_cpu_count'] = 0
_transfer_stats['cpu_to_gpu_bytes'] = 0
_transfer_stats['cpu_to_gpu_count'] = 0

# ---- Run eigsh ----
cp.get_default_memory_pool().free_all_blocks()
free_mem, total_mem = cp.cuda.runtime.memGetInfo()
if rank == 0:
    pool_used = mempool.used_bytes() / (1024**3)
    print(f"\n  [MEM] Before eigsh: pool={pool_used:.2f} GB, "
            f"free={free_mem/1024**3:.2f}/{total_mem/1024**3:.2f} GB")
    print("\n--- Running eigsh (Lanczos) ---")
    import sys; sys.stdout.flush()

with _nvtx_range("eigsh", color="green"):
    t_eigsh_start = time.perf_counter()
    ncv = min(20, subspace_dimension - 1)
    eigenvalues, timing, partition_info = distributed_eigsh(
        ham_data, rank, size, nccl_comm_obj,
        k=1, which='SA', ncv=ncv, v0_full=v0,
        tile_size=args.tile_size, return_eigenvectors=False)
    eigenvalue = eigenvalues[0]
    cp.cuda.Device(my_gpu).synchronize()
    t_eigsh_end = time.perf_counter()

# GPU memory after eigsh (peak from pool)
gpu_mem_peak_bytes = mempool.used_bytes()
gpu_mem_peak_mb = gpu_mem_peak_bytes / (1024 * 1024)

# Device-level peak memory (captures allocations outside CuPy pool)
free_mem_after, total_mem_dev = cp.cuda.runtime.memGetInfo()
device_peak_mem_mb = (total_mem_dev - free_mem_after) / (1024 * 1024)

# ---- GPU hardware state after eigsh (NVML) ----
nvml_after = {}
if _NVML_AVAILABLE and _nvml_handle is not None:
    try:
        nvml_after['sm_clock_mhz'] = pynvml.nvmlDeviceGetClockInfo(
            _nvml_handle, pynvml.NVML_CLOCK_SM)
        nvml_after['mem_clock_mhz'] = pynvml.nvmlDeviceGetClockInfo(
            _nvml_handle, pynvml.NVML_CLOCK_MEM)
        nvml_after['temperature_c'] = pynvml.nvmlDeviceGetTemperature(
            _nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
        nvml_after['power_mw'] = pynvml.nvmlDeviceGetPowerUsage(
            _nvml_handle)
        pynvml.nvmlShutdown()
    except Exception:
        pass

# Total wall time
t_wall_end = time.perf_counter()
total_wall_time = t_wall_end - t_wall_start

# ---- Collect and report results ----
total_eigsh_time = t_eigsh_end - t_eigsh_start
n_matvec = timing['n_calls']
total_compute = timing['compute_total']
total_comm = timing['comm_total']
total_matvec = timing['total']
total_lanczos_other = total_eigsh_time - total_matvec

avg_matvec = (total_matvec / n_matvec * 1000) if n_matvec > 0 else 0
avg_compute = (total_compute / n_matvec * 1000) if n_matvec > 0 else 0
avg_comm = (total_comm / n_matvec * 1000) if n_matvec > 0 else 0

# Matvec timing distribution
matvec_times_ms = np.array(timing['matvec_times']) * 1000
matvec_min_ms = float(np.min(matvec_times_ms)) if len(matvec_times_ms) > 0 else 0
matvec_max_ms = float(np.max(matvec_times_ms)) if len(matvec_times_ms) > 0 else 0
matvec_std_ms = float(np.std(matvec_times_ms)) if len(matvec_times_ms) > 0 else 0

# Convergence metrics
n_restarts = timing['n_restarts']
final_residual = timing['final_residual']
iter_count_total = timing['iter_count']
convergence_residuals = timing['convergence_residuals']

# Communication fraction (AllGather within matvec only, for comparability)
comm_fraction_pct = (timing['allgather_total'] / total_matvec * 100) if total_matvec > 0 else 0

# Effective GFLOP/s (upper bound: assumes all basis*terms pairs contribute)
# Per (row, term) pair: complex multiply-accumulate ~8 FLOPs
flops_per_matvec = subspace_dimension * n_terms * 8
total_flops = flops_per_matvec * n_matvec
effective_gflops = (total_flops / total_compute / 1e9) if total_compute > 0 else 0

# Effective memory bandwidth (GB/s)
# Per matvec: basis_ints + 6 term arrays + input vector (re+im) + output (re+im)
bytes_per_matvec = (
    subspace_dimension * 8
    + n_terms * 8 * 6
    + subspace_dimension * 8 * 2
    + partition_info['my_n_rows'] * 8 * 2
)
total_bytes_moved = bytes_per_matvec * n_matvec
effective_bw_gbs = (total_bytes_moved / total_compute / 1e9) if total_compute > 0 else 0

# Lanczos vector memory
lanczos_vec_mem_mb = (ncv * partition_info['chunk_size'] * 16) / (1024 * 1024)

# Hash table build time
hash_build_time = ham_data.get('hash_build_time', 0.0)

# Host peak RSS (Linux: ru_maxrss in KB)
host_peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

# Final GPU -> CPU transfer for eigenvalue
eigenvalue_np = cp.asnumpy(eigenvalue)
_transfer_stats['gpu_to_cpu_bytes'] += eigenvalue.nbytes
_transfer_stats['gpu_to_cpu_count'] += 1
eigenvalue_real = float(eigenvalue_np.real)

# Gather reference energies from JSON data
ref_energies = {}
for key, label in [('fci_energy', 'FCI'), ('hf_energy', 'HF'),
                    ('mp2_energy', 'MP2'), ('cisd_energy', 'CISD'),
                    ('ccsd_energy', 'CCSD'), ('ccsd(t)_energy', 'CCSD(T)')]:
    val = data.get(key)
    if val is not None:
        ref_energies[label] = val

# Eigenvalue accuracy metrics (vs best available reference)
best_ref_label, best_ref_val = None, None
for lbl in ['FCI', 'CCSD(T)', 'CCSD', 'CISD', 'MP2', 'HF']:
    if lbl in ref_energies:
        best_ref_label = lbl
        best_ref_val = ref_energies[lbl]
        break
if best_ref_val is not None:
    eigenvalue_rel_error = abs(eigenvalue_real - best_ref_val) / abs(best_ref_val)
    chem_accuracy_mha = abs(eigenvalue_real - best_ref_val) * 1000
else:
    eigenvalue_rel_error = float('nan')
    chem_accuracy_mha = float('nan')

# ---- NEW METRICS ----

# Communication breakdown by operation type
allgather_total = timing['allgather_total']
allgather_count = timing['allgather_count']
allreduce_total = timing['allreduce_total']
allreduce_count = timing['allreduce_count']
avg_allgather_ms = (allgather_total / allgather_count * 1000) if allgather_count > 0 else 0
avg_allreduce_ms = (allreduce_total / allreduce_count * 1000) if allreduce_count > 0 else 0

# Reorthogonalization cost
reorth_total = timing['reorth_total']
reorth_count = timing['reorth_count']
reorth_fraction_pct = (reorth_total / total_eigsh_time * 100) if total_eigsh_time > 0 else 0

# CUDA event-based kernel timing
kernel_event_times = np.array(timing['kernel_event_times_ms'])
if len(kernel_event_times) > 0:
    kernel_event_avg_ms = float(np.mean(kernel_event_times))
    kernel_event_min_ms = float(np.min(kernel_event_times))
    kernel_event_max_ms = float(np.max(kernel_event_times))
    kernel_event_std_ms = float(np.std(kernel_event_times))
else:
    kernel_event_avg_ms = 0
    kernel_event_min_ms = 0
    kernel_event_max_ms = 0
    kernel_event_std_ms = 0

# Warm-up vs steady-state separation
n_warmup = min(2, len(matvec_times_ms))
warmup_avg_ms = float(np.mean(matvec_times_ms[:n_warmup])) if n_warmup > 0 else 0
if len(matvec_times_ms) > n_warmup:
    steady_avg_ms = float(np.mean(matvec_times_ms[n_warmup:]))
    steady_std_ms = float(np.std(matvec_times_ms[n_warmup:]))
else:
    steady_avg_ms = warmup_avg_ms
    steady_std_ms = 0

# Communication bandwidth utilization
allgather_bytes_per_call = 2 * size * partition_info['chunk_size'] * 8
allgather_total_bytes = allgather_bytes_per_call * allgather_count
allgather_bw_gbs = (allgather_total_bytes / allgather_total / 1e9) if allgather_total > 0 else 0

# Communication-computation overlap ratio (0 = fully serial, no overlap)
sync_overhead = total_matvec - total_compute - allgather_total
overlap_ratio = 0.0
if total_matvec > 0:
    overlap_ratio = max(0.0, 1.0 - (total_compute + allgather_total) / total_matvec)

# Per-rank load imbalance (gather compute times from all ranks)
my_compute_arr = np.array([total_compute], dtype=np.float64)
my_n_rows_arr = np.array([partition_info['my_n_rows']], dtype=np.float64)
my_gpu_mem_arr = np.array([gpu_mem_peak_mb], dtype=np.float64)
if comm is not None and size > 1:
    all_computes = np.zeros(size, dtype=np.float64)
    all_n_rows = np.zeros(size, dtype=np.float64)
    all_gpu_mem = np.zeros(size, dtype=np.float64)
    comm.Gather(my_compute_arr, all_computes, root=0)
    comm.Gather(my_n_rows_arr, all_n_rows, root=0)
    comm.Gather(my_gpu_mem_arr, all_gpu_mem, root=0)
else:
    all_computes = my_compute_arr
    all_n_rows = my_n_rows_arr
    all_gpu_mem = my_gpu_mem_arr

if rank == 0:
    load_imbalance = float(np.max(all_computes) / np.mean(all_computes)) if np.mean(all_computes) > 0 else 1.0
    max_compute_rank = int(np.argmax(all_computes))
    min_compute_rank = int(np.argmin(all_computes))
else:
    load_imbalance = 1.0
    max_compute_rank = 0
    min_compute_rank = 0

# GPU hardware state
gpu_name = nvml_before.get('gpu_name', 'N/A')
sm_clock_before = nvml_before.get('sm_clock_mhz', -1)
sm_clock_after = nvml_after.get('sm_clock_mhz', -1)
mem_clock_before = nvml_before.get('mem_clock_mhz', -1)
mem_clock_after = nvml_after.get('mem_clock_mhz', -1)
temp_before = nvml_before.get('temperature_c', -1)
temp_after = nvml_after.get('temperature_c', -1)
power_before_w = nvml_before.get('power_mw', 0) / 1000.0
power_after_w = nvml_after.get('power_mw', 0) / 1000.0
avg_power_w = (power_before_w + power_after_w) / 2.0 if power_before_w > 0 else 0
energy_joules = avg_power_w * total_eigsh_time if avg_power_w > 0 else 0

# Kernel occupancy / register pressure
kern_num_regs = kernel_attrs.get('num_regs', -1)
kern_shared_bytes = kernel_attrs.get('shared_size_bytes', -1)
kern_max_tpb = kernel_attrs.get('max_threads_per_block', -1)

# ---- Cross-check with explicit Hamiltonian construction (rank 0) ----
# (commented out -- benchmark uses distributed_eigsh only)
# eigenvalue_ph = None
# eigenvalue_vph = None
# if rank == 0:
#     print(f"\n{'-' * 60}")
#     print(f"  CROSS-CHECK: explicit Hamiltonian construction")
#     print(f"{'-' * 60}")
#
#     t_ph_start = time.perf_counter()
#     ph = projected_hamiltonian(
#         basis_states, pauli_words, hamiltonian_coefficients_numpy)
#     eigenvalue_ph = scipy_eigsh(
#         ph, k=1, which='SA', return_eigenvectors=False)[0]
#     t_ph_end = time.perf_counter()
#     print(f"  projected_hamiltonian eigenvalue:  "
#           f"{float(eigenvalue_ph.real):.10f}  "
#           f"({t_ph_end - t_ph_start:.3f} s)")
#
#     t_vph_start = time.perf_counter()
#     vph = vectorized_projected_hamiltonian(
#         basis_states, pauli_words, hamiltonian_coefficients_numpy,
#         use_gpu=True)
#     eigenvalue_vph = float(cupy_eigsh(
#         vph, k=1, which='SA', return_eigenvectors=False)[0].real)
#     t_vph_end = time.perf_counter()
#     print(f"  vectorized_projected_hamiltonian:  "
#           f"{eigenvalue_vph:.10f}  "
#           f"({t_vph_end - t_vph_start:.3f} s)")
#
#     print(f"  distributed_eigsh eigenvalue:      "
#           f"{eigenvalue_real:.10f}")
#
#     diff_ph = abs(eigenvalue_real - float(eigenvalue_ph.real))
#     diff_vph = abs(eigenvalue_real - eigenvalue_vph)
#     print(f"  diff vs projected_hamiltonian:     {diff_ph:.2e}")
#     print(f"  diff vs vectorized_projected:      {diff_vph:.2e}")
#
#     del ph, vph
#     cp.get_default_memory_pool().free_all_blocks()

if rank == 0:
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Eigenvalue:        {eigenvalue_real:.10f}")
    if fci_energy is not None:
        print(f"  FCI energy:        {fci_energy:.10f}")
    else:
        print(f"  FCI energy:        N/A")
    print(f"  Total eigsh time:  {total_eigsh_time:.4f} s")
    print(f"  Matvec calls:      {n_matvec}")
    print(f"  Avg matvec time:   {avg_matvec:.3f} ms")
    print(f"    Compute:         {avg_compute:.3f} ms")
    print(f"    Communication:   {avg_comm:.3f} ms")
    print(f"  Total matvec:      {total_matvec:.4f} s "
            f"({total_matvec / total_eigsh_time * 100:.1f}%)")
    print(f"  Total compute:     {total_compute:.4f} s")
    print(f"  Total comm:        {total_comm:.4f} s")
    print(f"  Lanczos other:     {total_lanczos_other:.4f} s "
            f"({total_lanczos_other / total_eigsh_time * 100:.1f}%)")
    print(f"  Setup time:        {t_setup_end - t_setup_start:.4f} s")
    print(f"  GPUs:              {size}")
    print(f"-" * 60)
    print(f"  MATVEC DISTRIBUTION (host timers)")
    print(f"  Min matvec:           {matvec_min_ms:.3f} ms")
    print(f"  Max matvec:           {matvec_max_ms:.3f} ms")
    print(f"  Stddev matvec:        {matvec_std_ms:.3f} ms")
    print(f"  Comm fraction:        {comm_fraction_pct:.2f}%")
    print(f"-" * 60)
    print(f"  CUDA EVENT KERNEL TIMING")
    print(f"  Avg kernel (event):   {kernel_event_avg_ms:.3f} ms")
    print(f"  Min kernel (event):   {kernel_event_min_ms:.3f} ms")
    print(f"  Max kernel (event):   {kernel_event_max_ms:.3f} ms")
    print(f"  Std kernel (event):   {kernel_event_std_ms:.3f} ms")
    print(f"-" * 60)
    print(f"  WARM-UP vs STEADY-STATE")
    print(f"  Warm-up avg ({n_warmup} calls): {warmup_avg_ms:.3f} ms")
    print(f"  Steady avg:           {steady_avg_ms:.3f} ms")
    print(f"  Steady stddev:        {steady_std_ms:.3f} ms")
    print(f"-" * 60)
    print(f"  COMMUNICATION BREAKDOWN")
    print(f"  AllGather total:      {allgather_total:.4f} s  ({allgather_count} calls)")
    print(f"  AllGather avg:        {avg_allgather_ms:.3f} ms")
    print(f"  AllReduce total:      {allreduce_total:.4f} s  ({allreduce_count} calls)")
    print(f"  AllReduce avg:        {avg_allreduce_ms:.3f} ms")
    print(f"  AllGather BW:         {allgather_bw_gbs:.2f} GB/s")
    print(f"  Overlap ratio:        {overlap_ratio:.4f}")
    print(f"-" * 60)
    print(f"  REORTHOGONALIZATION")
    print(f"  Reorth total:         {reorth_total:.4f} s  ({reorth_count} calls)")
    print(f"  Reorth fraction:      {reorth_fraction_pct:.2f}%")
    print(f"-" * 60)
    print(f"  LOAD BALANCE")
    print(f"  Load imbalance:       {load_imbalance:.4f} (max/mean compute)")
    print(f"  Slowest rank:         {max_compute_rank}  ({all_computes[max_compute_rank]:.4f} s)")
    print(f"  Fastest rank:         {min_compute_rank}  ({all_computes[min_compute_rank]:.4f} s)")
    if size > 1:
        print(f"  Rows per rank:        {[int(r) for r in all_n_rows]}")
    print(f"-" * 60)
    print(f"  CONVERGENCE")
    print(f"  Lanczos iters:        {iter_count_total}")
    print(f"  Thick restarts:       {n_restarts}")
    print(f"  Final residual:       {final_residual:.6e}")
    print(f"  Residual history:     {['%.2e' % r for r in convergence_residuals]}")
    print(f"-" * 60)
    print(f"  THROUGHPUT")
    print(f"  Effective GFLOPS:     {effective_gflops:.2f}")
    print(f"  Effective BW GBs:     {effective_bw_gbs:.2f}")
    print(f"-" * 60)
    print(f"  MEMORY")
    print(f"  GPU mem after setup:  {gpu_mem_after_setup_mb:.1f} MB")
    print(f"  GPU mem peak:         {gpu_mem_peak_mb:.1f} MB")
    print(f"  Dev mem peak:         {device_peak_mem_mb:.1f} MB")
    print(f"  Lanczos vec mem:      {lanczos_vec_mem_mb:.1f} MB")
    print(f"  Host peak RSS:        {host_peak_rss_mb:.1f} MB")
    print(f"  Hash build time:      {hash_build_time:.4f} s")
    print(f"-" * 60)
    print(f"  DATA TRANSFERS")
    print(f"  Ritz solve calls:     {_transfer_stats['ritz_calls']}")
    print(f"  Ritz solve time:      {_transfer_stats['ritz_time']:.4f} s")
    print(f"  GPU->CPU transfers:   {_transfer_stats['gpu_to_cpu_count']}")
    print(f"  GPU->CPU bytes:       {_transfer_stats['gpu_to_cpu_bytes']}")
    print(f"  CPU->GPU transfers:   {_transfer_stats['cpu_to_gpu_count']}")
    print(f"  CPU->GPU bytes:       {_transfer_stats['cpu_to_gpu_bytes']}")
    print(f"-" * 60)
    print(f"  GPU HARDWARE STATE")
    print(f"  GPU name:             {gpu_name}")
    print(f"  SM clock (before):    {sm_clock_before} MHz")
    print(f"  SM clock (after):     {sm_clock_after} MHz")
    print(f"  Mem clock (before):   {mem_clock_before} MHz")
    print(f"  Mem clock (after):    {mem_clock_after} MHz")
    print(f"  Temp (before):        {temp_before} C")
    print(f"  Temp (after):         {temp_after} C")
    print(f"  Power (before):       {power_before_w:.1f} W")
    print(f"  Power (after):        {power_after_w:.1f} W")
    print(f"  Energy (est):         {energy_joules:.1f} J")
    print(f"-" * 60)
    print(f"  KERNEL ATTRIBUTES")
    print(f"  Registers/thread:     {kern_num_regs}")
    print(f"  Shared mem/block:     {kern_shared_bytes} B")
    print(f"  Max threads/block:    {kern_max_tpb}")
    print(f"-" * 60)
    print(f"  EIGENVALUE COMPARISON")
    for label, ref_val in ref_energies.items():
        diff = eigenvalue_real - ref_val
        print(f"    vs {label:>8s}: {ref_val:.10f}  diff: {diff:+.10f}")
    if best_ref_val is not None:
        print(f"  Rel error:            {eigenvalue_rel_error:.6e}")
        print(f"  Error mHa:            {chem_accuracy_mha:.4f}")
        print(f"  Chem accuracy:        {'YES' if chem_accuracy_mha <= 1.6 else 'NO'} (threshold: 1.6 mHa)")
    print(f"-" * 60)
    print(f"  Total wall time:      {total_wall_time:.4f} s")
    print(f"{'=' * 60}")

    # ---- Save results to CSV (rank 0 only) ----
    mol_stem = os.path.splitext(os.path.basename(args.molecule))[0]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    csv_path = args.output_csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                csv_path)

    csv_header = (
        "timestamp,molecule,n_gpus,n_configs,total_time_s,setup_time_s,"
        "total_matvec_s,total_compute_s,total_comm_s,lanczos_other_s,"
        "avg_matvec_ms,avg_compute_ms,avg_comm_ms,matvec_calls,"
        "gpu_mem_setup_mb,gpu_mem_peak_mb,ritz_calls,ritz_time_s,"
        "gpu2cpu_transfers,gpu2cpu_bytes,cpu2gpu_transfers,cpu2gpu_bytes,"
        "eigenvalue,n_restarts,iter_count,final_residual,"
        "matvec_min_ms,matvec_max_ms,matvec_stddev_ms,"
        "effective_gflops,effective_bw_gbs,hash_build_time_s,"
        "comm_fraction_pct,device_peak_mem_mb,host_peak_rss_mb,"
        "total_wall_time_s,lanczos_vec_mem_mb,rel_error,error_mha,"
        "kernel_event_avg_ms,kernel_event_min_ms,kernel_event_max_ms,"
        "kernel_event_std_ms,"
        "allgather_total_s,allgather_count,allreduce_total_s,allreduce_count,"
        "avg_allgather_ms,avg_allreduce_ms,allgather_bw_gbs,"
        "reorth_total_s,reorth_count,reorth_fraction_pct,"
        "load_imbalance,max_compute_rank,min_compute_rank,"
        "warmup_avg_ms,steady_avg_ms,steady_std_ms,"
        "overlap_ratio,"
        "sm_clock_before_mhz,sm_clock_after_mhz,"
        "mem_clock_before_mhz,mem_clock_after_mhz,"
        "temp_before_c,temp_after_c,"
        "power_before_w,power_after_w,energy_j,"
        "kern_num_regs,kern_shared_bytes,kern_max_tpb,"
        "gpu_name"
    )

    csv_row = (
        f"{timestamp},{mol_stem},{size},{n_configs},"
        f"{total_eigsh_time:.4f},"
        f"{t_setup_end - t_setup_start:.4f},"
        f"{total_matvec:.4f},{total_compute:.4f},{total_comm:.4f},"
        f"{total_lanczos_other:.4f},"
        f"{avg_matvec:.3f},{avg_compute:.3f},{avg_comm:.3f},{n_matvec},"
        f"{gpu_mem_after_setup_mb:.1f},{gpu_mem_peak_mb:.1f},"
        f"{_transfer_stats['ritz_calls']},"
        f"{_transfer_stats['ritz_time']:.4f},"
        f"{_transfer_stats['gpu_to_cpu_count']},"
        f"{_transfer_stats['gpu_to_cpu_bytes']},"
        f"{_transfer_stats['cpu_to_gpu_count']},"
        f"{_transfer_stats['cpu_to_gpu_bytes']},"
        f"{eigenvalue_real:.10f},{n_restarts},{iter_count_total},"
        f"{final_residual:.6e},"
        f"{matvec_min_ms:.3f},{matvec_max_ms:.3f},{matvec_std_ms:.3f},"
        f"{effective_gflops:.2f},{effective_bw_gbs:.2f},"
        f"{hash_build_time:.4f},"
        f"{comm_fraction_pct:.2f},{device_peak_mem_mb:.1f},"
        f"{host_peak_rss_mb:.1f},"
        f"{total_wall_time:.4f},{lanczos_vec_mem_mb:.1f},"
        f"{eigenvalue_rel_error:.6e},{chem_accuracy_mha:.4f},"
        f"{kernel_event_avg_ms:.3f},{kernel_event_min_ms:.3f},"
        f"{kernel_event_max_ms:.3f},{kernel_event_std_ms:.3f},"
        f"{allgather_total:.4f},{allgather_count},"
        f"{allreduce_total:.4f},{allreduce_count},"
        f"{avg_allgather_ms:.3f},{avg_allreduce_ms:.3f},"
        f"{allgather_bw_gbs:.2f},"
        f"{reorth_total:.4f},{reorth_count},{reorth_fraction_pct:.2f},"
        f"{load_imbalance:.4f},{max_compute_rank},{min_compute_rank},"
        f"{warmup_avg_ms:.3f},{steady_avg_ms:.3f},{steady_std_ms:.3f},"
        f"{overlap_ratio:.4f},"
        f"{sm_clock_before},{sm_clock_after},"
        f"{mem_clock_before},{mem_clock_after},"
        f"{temp_before},{temp_after},"
        f"{power_before_w:.1f},{power_after_w:.1f},{energy_joules:.1f},"
        f"{kern_num_regs},{kern_shared_bytes},{kern_max_tpb},"
        f"{gpu_name}"
    )

    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as csv_f:
        if write_header:
            csv_f.write(csv_header + '\n')
        csv_f.write(csv_row + '\n')
    print(f"\n  Results appended to: {csv_path}")

if _MPI_AVAILABLE:
    MPI.Finalize()
