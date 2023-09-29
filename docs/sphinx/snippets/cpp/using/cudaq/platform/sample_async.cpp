// Compile and run with:
/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// ```
// nvq++ sample_async.cpp -o sample_async.x -target nvidia-mqpu
// && ./sample_async.x
// ```
#include <cudaq.h>

int main() {
  // [Begin Documentation]
  auto kernelToBeSampled = [](int runtimeParam) __qpu__ {
    cudaq::qreg q(runtimeParam);
    h(q);
    mz(q);
  };

  // Get the quantum_platform singleton
  auto &platform = cudaq::get_platform();

  // Query the number of QPUs in the system
  auto num_qpus = platform.num_qpus();
  printf("Number of QPUs: %zu\n", num_qpus);
  // We will launch asynchronous sampling tasks
  // and will store the results immediately as a future
  // we can query at some later point
  std::vector<cudaq::async_sample_result> countFutures;
  for (std::size_t i = 0; i < num_qpus; i++) {
    countFutures.emplace_back(
        cudaq::sample_async(i, kernelToBeSampled, 5 /*runtimeParam*/));
  }

  //
  // Go do other work, asynchronous execution of sample tasks on-going
  //

  // Get the results, note future::get() will kick off a wait
  // if the results are not yet available.
  for (auto &counts : countFutures) {
    counts.get().dump();
  }
  // [End Documentation]
  return 0;
}
