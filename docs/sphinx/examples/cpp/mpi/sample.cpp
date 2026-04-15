/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This example demonstrates how to use multiple MPI-based simulators backend in
// CUDA-Q. Specifically, each simulator instance will use 2 ranks (GPUs).

// For example, to use the tensornet backend:
//
// nvq++ --target tensornet -o sample
// sample.cpp mpirun -n 4 ./sample
//

#include <cudaq.h>

__qpu__ void ghz(int count) {
  cudaq::qvector v(count);
  h(v[0]);
  for (int i = 0; i < count - 1; i++)
    cx(v[i], v[i + 1]);
}

int main() {
  cudaq::mpi::initialize();

  int num_ranks = cudaq::mpi::num_ranks();

  if (num_ranks % 2 != 0) {
    if (cudaq::mpi::rank() == 0)
      std::cerr << "This test requires an even number of MPI ranks."
                << std::endl;
    cudaq::mpi::finalize();
    return 1;
  }

  int num_qpus = num_ranks / 2;
  int qpu_color =
      cudaq::mpi::rank() /
      2; // Each pair of ranks will form a color for the split communicator.
  auto comm_ptr = cudaq::mpi::split_communicator(
      qpu_color); // Split the global communicator into sub-communicators for
                  // each QPU.
  cudaq::mpi::set_communicator(
      comm_ptr); // Set the communicator for the current rank's QPU.

  for (int qpu = 0; qpu < num_qpus; qpu++) {
    if (qpu_color == qpu) {
      const auto num_qubits =
          20 + qpu; // Dispatch different number of qubits to each backend.
      std::cout << "Rank " << cudaq::mpi::rank() << " is in color " << qpu_color
                << " for QPU " << qpu << " with " << num_qubits << " qubits."
                << std::endl;

      auto results = cudaq::sample(ghz, num_qubits);
      std::cout << "Results from rank " << cudaq::mpi::rank() << " for QPU "
                << qpu << ": \n";
      results.dump();
    }
  }

  cudaq::mpi::finalize();
  return 0;
}
