/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: mpi

// clang-format off
// RUN: nvq++ --target qpp-cpu --platform mqpu %s -o %t && mpiexec -np 4 --allow-run-as-root %t 
// RUN: nvq++ --target qpp-cpu --platform mqpu %s -o %t && mpiexec -np 4 --allow-run-as-root %t
// clang-format on

#include <cudaq.h>

__qpu__ int test_kernel(int count) {
  unsigned result = 0;
  cudaq::qvector v(count);
  h(v[0]);
  for (int i = 0; i < count - 1; i++)
    cx(v[i], v[i + 1]);
  for (int i = 0; i < count; i++)
    if (mz(v[i]))
      result += 1;
  return result;
}

int main() {
  cudaq::mpi::initialize();

  // Get the quantum_platform singleton
  auto &platform = cudaq::get_platform();

  // Query the number of QPUs in the system
  auto num_qpus = platform.num_qpus();
  printf("Number of QPUs: %zu\n", num_qpus);

  assert(num_qpus == cudaq::mpi::num_ranks() &&
         "Number of QPUs must match number of MPI ranks for this test.");

  for (int qpu = 0; qpu < num_qpus; qpu++) {
    auto results = cudaq::run_async(qpu, 100, test_kernel, 2 + qpu).get();
    if (cudaq::mpi::rank() == qpu) {
      assert(results.size() == 100 && "Expected 100 results from run.");
      for (const auto &result : results) {
        assert(result == 0 ||
               result == (2 + qpu) &&
                   "Expected result to be either all 0s or number of qubits.");
      }
    }
  }

  cudaq::mpi::finalize();
  return 0;
}
