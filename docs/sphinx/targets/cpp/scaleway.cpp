/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ --target scaleway scaleway.cpp -o out.x && ./out.x
// ```
// This will submit the job to the Scaleway QaaS state vector emulator powered by CUDA-Q
// (default). Alternatively, users can choose any of the available devices by
// specifying its name with the `--machine`, e.g.,
// ```
// nvq++ --target scaleway --machine \
// "EMU-CUDAQ-H100" scaleway.cpp -o out.x
// ./out.x
// ```
// Assumes a valid set of credentials have been set prior to execution.

#include <cudaq.h>
#include <fstream>

// Define a simple quantum kernel to execute on Scaleway's QaaS.
struct ghz {
  // Maximally entangled state between 5 qubits.
  auto operator()() __qpu__ {
    cudaq::qvector q(5);
    h(q[0]);
    for (int i = 0; i < 4; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    auto result = mz(q);
  }
};

int main() {
  // Submit asynchronously (e.g., continue executing
  // code in the file until the job has been returned).
  auto future = cudaq::sample_async(ghz{});
  // ... classical code to execute in the meantime ...

  // Get the results of the read in future.
  auto async_counts = future.get();
  async_counts.dump();

  // OR: Submit synchronously (e.g., wait for the job
  // result to be returned before proceeding).
  auto counts = cudaq::sample(ghz{});
  counts.dump();
}
