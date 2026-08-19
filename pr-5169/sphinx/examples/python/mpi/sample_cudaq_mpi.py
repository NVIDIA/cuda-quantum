# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Demonstrates multi-node multi-GPU simulation using the `cudaq.mpi` API to
# partition ranks into independent QPU groups, each backed by a multi-GPU
# simulator (e.g. `tensornet`). Every group simulates a different circuit.
#
# Run (4 ranks → 2 QPUs of 2 ranks/GPUs each):
# ```
#   mpirun -n 4 python3 sample_cudaq_mpi.py
# ```

# [Begin Documentation]
import cudaq


@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        cx(q[i], q[i + 1])


cudaq.mpi.initialize()

world_rank = cudaq.mpi.rank()
world_size = cudaq.mpi.num_ranks()

# Each QPU is backed by `ranks_per_qpu` MPI ranks / GPUs.
ranks_per_qpu = 2
if world_size % ranks_per_qpu != 0:
    if world_rank == 0:
        print(f"World size must be a multiple of {ranks_per_qpu}.")
    cudaq.mpi.finalize()
    raise SystemExit(1)

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
qpu_comm = cudaq.mpi.split_communicator(qpu_id)

# Select the target and pass the sub-communicator for this QPU group.
cudaq.set_target("tensornet", comm_handle=qpu_comm)

# Run an independent circuit on each QPU group.
# The `tensornet` backend can handle a large number of qubits via multi-GPU
# tensor network contraction, well beyond what state vector simulators support.
num_qubits = 40 + 5 * qpu_id
result = cudaq.sample(ghz, num_qubits)

if world_rank % ranks_per_qpu == 0:
    print(f"QPU {qpu_id} ({num_qubits} qubits):")
    print(result)

cudaq.mpi.finalize()
# [End Documentation]
