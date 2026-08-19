/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Demonstrates multi-node multi-GPU simulation using the `cudaq::mpi` API to
// partition ranks into independent QPU groups, each backed by a multi-GPU
// simulator (e.g. `tensornet`). Every group simulates a different circuit.
//
// Build and run (4 ranks → 2 QPUs of 2 ranks/GPUs each):
// ```
//   nvq++ --target tensornet -o sample_cudaq_mpi sample_cudaq_mpi.cpp
//   mpirun -n 4 ./sample_cudaq_mpi
// ```
//
// Note: no explicit MPI flags needed; `nvq++` links MPI automatically.

// [Begin Documentation]
#include <cudaq.h>

__qpu__ void ghz(int n) {
  cudaq::qvector q(n);
  h(q[0]);
  for (int i = 0; i < n - 1; i++)
    cx(q[i], q[i + 1]);
}

int main() {
  cudaq::mpi::initialize();

  const int world_rank = cudaq::mpi::rank();
  const int world_size = cudaq::mpi::num_ranks();

  // Each `QPU` is backed by `ranks_per_qpu` MPI ranks / GPUs.
  const int ranks_per_qpu = 2;
  if (world_size % ranks_per_qpu != 0) {
    if (world_rank == 0)
      fprintf(stderr, "World size must be a multiple of %d.\n", ranks_per_qpu);
    cudaq::mpi::finalize();
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
  void *qpu_comm = cudaq::mpi::split_communicator(qpu_id);

  // Inform CUDA-Q which sub-communicator this `QPU` group should use.
  cudaq::mpi::set_communicator(qpu_comm);

  // Run an independent circuit on each `QPU` group.
  // The `tensornet` backend can handle a large number of qubits via multi-GPU
  // tensor network contraction, well beyond what state vector simulators
  // support.
  const int num_qubits = 40 + 5 * qpu_id;
  auto result = cudaq::sample(ghz, num_qubits);

  if (world_rank % ranks_per_qpu == 0) {
    printf("QPU %d (%d qubits):\n", qpu_id, num_qubits);
    result.dump();
  }

  cudaq::mpi::finalize();
  return 0;
}
// [End Documentation]
