/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ sample_async_remote.cpp -o sample.x \
//   --target remote-mqpu --remote-mqpu-auto-launch 2
// ./sample.x
// ```
#include <cudaq.h>

int main() {
  // [Begin Documentation]
  // Define a kernel to be sampled.
  auto [kernel, nrControls] = cudaq::make_kernel<int>();
  auto controls = kernel.qalloc(nrControls);
  auto targets = kernel.qalloc(2);
  kernel.h(controls);
  for (std::size_t tidx = 0; tidx < 2; ++tidx) {
    kernel.x<cudaq::ctrl>(controls, targets[tidx]);
  }
  kernel.mz(controls);
  kernel.mz(targets);

  // Query the number of QPUs in the system;
  // The number of QPUs is equal to the number of (auto-)launched server
  // instances.
  auto &platform = cudaq::get_platform();
  auto num_qpus = platform.num_qpus();
  printf("Number of QPUs: %zu\n", num_qpus);

  // We will launch asynchronous sampling tasks,
  // and will store the results as a future we can query at some later point.
  // Each QPU (indexed by an unique Id) is associated with a remote REST server.
  std::vector<cudaq::async_sample_result> countFutures;
  for (std::size_t i = 0; i < num_qpus; i++) {
    countFutures.emplace_back(cudaq::sample_async(
        /*qpuId=*/i, kernel, /*nrControls=*/i + 1));
  }

  // Go do other work, asynchronous execution of sample tasks on-going
  // Get the results, note future::get() will kick off a wait
  // if the results are not yet available.
  for (auto &counts : countFutures) {
    counts.get().dump();
  }
  // [End Documentation]
  return 0;
}
