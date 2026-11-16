/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Demonstrates multi-node multi-GPU simulation by partitioning MPI ranks into
// independent QPU groups, each backed by a multi-GPU simulator (e.g.
// `tensornet`). Every group simulates a different circuit in parallel.
//
// Build and run (4 ranks → 2 QPUs of 2 ranks/GPUs each):
// clang-format off
// ```
// nvq++ --target tensornet -o sample sample.cpp -I$(mpicc -showme:incdirs) -L$(mpicc -showme:libdirs) -lmpi
// mpirun -n 4 ./sample
// ```
// clang-format on

// [Begin Documentation]
#include <cudaq.h>
#include <mpi.h>

__qpu__ void ghz(int n) {
  cudaq::qvector q(n);
  h(q[0]);
  for (int i = 0; i < n - 1; i++)
    cx(q[i], q[i + 1]);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Each `QPU` is backed by `ranks_per_qpu` MPI ranks / GPUs.
  const int ranks_per_qpu = 2;
  if (world_size % ranks_per_qpu != 0) {
    if (world_rank == 0)
      fprintf(stderr, "World size must be a multiple of %d.\n", ranks_per_qpu);
    MPI_Finalize();
    return 1;
  }

  // Assign each rank to a `QPU` group and split the communicator accordingly.
  // ```
  //  MPI_COMM_WORLD
  //  +----------+----------+----------+----------+
  //  |  rank 0  |  rank 1  |  rank 2  |  rank 3  |
  //  +----------+----------+----------+----------+
  //  |        QPU 0        |        QPU 1        |
  //  |    (qpu_id = 0)     |    (qpu_id = 1)     |
  //  +---------------------+---------------------+
  // ```
  const int qpu_id = world_rank / ranks_per_qpu;
  MPI_Comm qpu_comm;
  MPI_Comm_split(MPI_COMM_WORLD, qpu_id, world_rank, &qpu_comm);

  // Inform CUDA-Q which sub-communicator this `QPU` group should use.
  cudaq::mpi::set_communicator(reinterpret_cast<void *>(&qpu_comm));

  // Run an independent circuit on each `QPU` group.
  // The `tensornet` backend can handle a large number of qubits via multi-GPU
  // tensor network contraction, well beyond what state vector simulators
  // support.
  const int num_qubits = 40 + 5 * qpu_id;
  auto result = cudaq::sample(ghz, num_qubits);

  int qpu_rank;
  MPI_Comm_rank(qpu_comm, &qpu_rank);
  if (qpu_rank == 0) {
    printf("QPU %d (%d qubits):\n", qpu_id, num_qubits);
    result.dump();
  }

  MPI_Comm_free(&qpu_comm);
  MPI_Finalize();
  return 0;
}
// [End Documentation]
