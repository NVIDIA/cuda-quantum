/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ get_state_async.cpp -o get_state_async.x -target nvidia-mqpu
// && ./get_state_async.x
// ```
#include <cudaq.h>
#include <cudaq/algorithms/get_state.h>
int main() {
  // [Begin Documentation]
  auto kernelToRun = [](int runtimeParam) __qpu__ {
    cudaq::qvector q(runtimeParam);
    h(q[0]);
    for (int i = 0; i < runtimeParam - 1; ++i)
      x<cudaq::ctrl>(q[i], q[i + 1]);
  };

  // Get the quantum_platform singleton
  auto &platform = cudaq::get_platform();

  // Query the number of QPUs in the system
  auto num_qpus = platform.num_qpus();
  printf("Number of QPUs: %zu\n", num_qpus);
  // We will launch asynchronous tasks
  // and will store the results immediately as a future
  // we can query at some later point
  std::vector<cudaq::async_state_result> stateFutures;
  for (std::size_t i = 0; i < num_qpus; i++) {
    stateFutures.emplace_back(
        cudaq::get_state_async(i, kernelToRun, 5 /*runtimeParam*/));
  }

  //
  // Go do other work, asynchronous execution of tasks on-going
  //

  // Get the results, note future::get() will kick off a wait
  // if the results are not yet available.
  for (auto &state : stateFutures) {
    state.get().dump();
  }
  // [End Documentation]
  return 0;
}
