# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Demonstrates gradient computation (parameter-shift rule) where each QPU
# group runs a multi-GPU observe() call for its assigned gradient component.
#
# Each observe() is distributed across `ranks_per_qpu` GPUs via the MPI
# sub-communicator, enabling circuits too large for one GPU.
#
# Run (4 ranks → 2 QPUs of 2 ranks/GPUs each):
# ```
#   mpirun -n 4 python3 observe_gradient.py
# ```

# [Begin Documentation]
import math
import sys

import cudaq
from mpi4py import MPI


@cudaq.kernel
def ansatz(n: int, theta0: float, theta1: float):
    """Dummy ansatz for demonstration: replace with your VQE circuit.
    Ry(theta0) on even qubits, Ry(theta1) on odd qubits — product state,
    no entanglement. With H = sum_i Z(i) this gives an analytically
    verifiable gradient: grad[k] = -n/2 * sin(theta_k).
    """
    q = cudaq.qvector(n)
    for i in range(n):
        if i % 2 == 0:
            ry(theta0, q[i])
        else:
            ry(theta1, q[i])


def mpi_comm_handle(comm) -> int:
    """Return a pointer to the MPI_Comm handle as an integer."""
    return MPI._addressof(comm)


world_comm = MPI.COMM_WORLD
world_rank = world_comm.Get_rank()
world_size = world_comm.Get_size()

ranks_per_qpu = 2
if world_size % ranks_per_qpu != 0:
    if world_rank == 0:
        print(f"World size must be a multiple of {ranks_per_qpu}.",
              file=sys.stderr)
    sys.exit(1)

num_qpus = world_size // ranks_per_qpu

# Assign each rank to a QPU group and split the communicator.
# QPU group g computes the gradient for parameter theta_g.
#
# This example uses 2 parameters and 2 QPU groups. To scale to N parameters,
# launch with N * `ranks_per_qpu` total MPI ranks and provide N initial `params`.
# Each additional QPU group adds `ranks_per_qpu` ranks and handles one more
# gradient component in parallel — no other code changes required.
#
#  MPI_COMM_WORLD
#  +----------+----------+----------+----------+
#  |  rank 0  |  rank 1  |  rank 2  |  rank 3  |
#  +----------+----------+----------+----------+
#  |        QPU 0        |        QPU 1        |
#  | grad[0]=(E+-E-)/2   | grad[1]=(E+-E-)/2   |
#  +---------------------+---------------------+
#
qpu_id = world_rank // ranks_per_qpu
qpu_comm = world_comm.Split(color=qpu_id, key=world_rank)

# Each QPU group uses `ranks_per_qpu` GPUs for every cudaq.observe() call.
cudaq.set_target("tensornet", comm_handle=mpi_comm_handle(qpu_comm))

# Dummy Hamiltonian (sum of Z on all qubits) for demonstration:
# replace with your physical Hamiltonian, e.g. generated from PySCF.
n_qubits = 40
H = cudaq.spin.z(0)
for i in range(1, n_qubits):
    H += cudaq.spin.z(i)
params = [0.5, 0.3]  # theta_0, theta_1
shift = math.pi / 2.0

# Each QPU group applies +/-shift to its assigned parameter and runs two
# observe() calls. Both calls use all `ranks_per_qpu` GPUs via the
# sub-communicator, enabling multi-GPU tensor-network contraction.
#
# In a VQE optimization loop, this block executes every iteration:
# the optimizer supplies updated `params`, each QPU group re-evaluates its
# two shifted energies, and the gathered gradient drives the next step.
p_plus = params.copy()
p_plus[qpu_id] += shift
p_minus = params.copy()
p_minus[qpu_id] -= shift

e_plus = cudaq.observe(ansatz, H, n_qubits, p_plus[0], p_plus[1]).expectation()
e_minus = cudaq.observe(ansatz, H, n_qubits, p_minus[0],
                        p_minus[1]).expectation()
local_grad = (e_plus - e_minus) / 2.0

# Gather local_grad from every world rank. All ranks within a QPU group
# hold the same value (collective result), so sample every `ranks_per_qpu-th`
# entry to reconstruct the gradient vector.
all_grads = world_comm.allgather(local_grad)

if world_rank == 0:
    gradient = [all_grads[i * ranks_per_qpu] for i in range(num_qpus)]
    print(f"Gradient: {gradient}")

qpu_comm.Free()
# [End Documentation]
