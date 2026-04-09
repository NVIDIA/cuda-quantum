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
// clang-format on

#include <cudaq.h>

__qpu__ void ghz(int count) {
  cudaq::qvector v(count);
  h(v[0]);
  for (int i = 0; i < count - 1; i++)
    cx(v[i], v[i + 1]);
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
    auto state = cudaq::get_state_async(qpu, ghz, 2 + qpu).get();
    const std::vector<int> expectedState1(2 + qpu, 0);
    const std::vector<int> expectedState2(2 + qpu, 1);
    if (cudaq::mpi::rank() == qpu) {
      assert(state.get_num_qubits() == 2 + qpu &&
             "Unexpected number of qubits in the GHZ state.");
      assert(std::abs(std::norm(state.amplitude(expectedState1)) - 0.5) <
                 1e-3 &&
             "Expected non-zero amplitude for the |00...0> state.");
      assert(std::abs(std::norm(state.amplitude(expectedState2)) - 0.5) <
                 1e-3 &&
             "Expected non-zero amplitude for the |11...1> state.");
    }
  }

  cudaq::mpi::finalize();
  return 0;
}
