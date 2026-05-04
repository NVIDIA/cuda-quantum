# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Demonstrates multi-node multi-GPU simulation by partitioning MPI ranks into
# independent QPU groups using `mpi4py`, each backed by a multi-GPU simulator
# (e.g. `tensornet`). Every group simulates a different circuit in parallel.
#
# Run (4 ranks → 2 QPUs of 2 ranks/GPUs each):
# ```
#   mpirun -n 4 python3 sample.py
# ```

# [Begin Documentation]
import sys

import cudaq
from mpi4py import MPI


@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        cx(q[i], q[i + 1])


def mpi_comm_handle(comm) -> int:
    """Return a pointer to the MPI_Comm handle as an integer."""
    return MPI._addressof(comm)


world_comm = MPI.COMM_WORLD
world_rank = world_comm.Get_rank()
world_size = world_comm.Get_size()

# Each QPU is backed by `ranks_per_qpu` MPI ranks / GPUs.
ranks_per_qpu = 2
if world_size % ranks_per_qpu != 0:
    if world_rank == 0:
        print(f"World size must be a multiple of {ranks_per_qpu}.",
              file=sys.stderr)
    sys.exit(1)

# Assign each rank to a QPU group and split the communicator accordingly.
# ```
#  MPI_COMM_WORLD
#  +----------+----------+----------+----------+
#  |  rank 0  |  rank 1  |  rank 2  |  rank 3  |
#  +----------+----------+----------+----------+
#  |        QPU 0        |        QPU 1        |
#  |    (qpu_id = 0)     |    (qpu_id = 1)     |
#  +---------------------+---------------------+
# ```
qpu_id = world_rank // ranks_per_qpu
qpu_comm = world_comm.Split(color=qpu_id, key=world_rank)

# Pass the sub-communicator handle to CUDA-Q when selecting the target.
cudaq.set_target("tensornet", comm_handle=mpi_comm_handle(qpu_comm))

# Run an independent circuit on each QPU group.
# The `tensornet` backend can handle a large number of qubits via multi-GPU
# tensor network contraction, well beyond what state vector simulators support.
num_qubits = 40 + 5 * qpu_id
result = cudaq.sample(ghz, num_qubits)

if qpu_comm.Get_rank() == 0:
    print(f"QPU {qpu_id} ({num_qubits} qubits):")
    print(result)
# [End Documentation]
