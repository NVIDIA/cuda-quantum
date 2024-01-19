/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ sample_async.cpp -o sample_async.x --target remote-mqpu
// --remote-mqpu-auto-launch 2
// && ./sample_async.x
// ```
#include <cudaq.h>

int main() {
  // [Begin Documentation]
  auto [kernelToBeSampled, runtimeParam] = cudaq::make_kernel<int>();
  auto q = kernelToBeSampled.qalloc(runtimeParam);
  kernelToBeSampled.h(q);
  kernelToBeSampled.mz(q);

  // Get the quantum_platform singleton
  auto &platform = cudaq::get_platform();

  // Query the number of QPUs in the system
  // The number of QPUs is equal to the number of auto-launch server instances.
  auto num_qpus = platform.num_qpus();
  printf("Number of QPUs: %zu\n", num_qpus);
  // We will launch asynchronous sampling tasks
  // and will store the results immediately as a future
  // we can query at some later point.
  // Each QPU (indexed by an unique Id) is associated with a remote REST server.
  std::vector<cudaq::async_sample_result> countFutures;
  for (std::size_t i = 0; i < num_qpus; i++) {
    countFutures.emplace_back(cudaq::sample_async(
        /*qpuId=*/i, kernelToBeSampled, /*runtimeParam=*/5));
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
