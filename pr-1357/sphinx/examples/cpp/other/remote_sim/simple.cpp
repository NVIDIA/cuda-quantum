/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// clang-format off
// ```
//  nvq++ --target remote-mqpu --remote-mqpu-auto-launch 1 simple.cpp -o simple.x &&
// ./simple.x
// ```
// clang-format on

#include <cudaq.h>
#include <cudaq/algorithm.h>

// The example here shows a simple use case for the remote simulator platform
// (`remote-mqpu`). Please refer to the documentation for more information about
// its features and command line options.

int main() {
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto [ansatz, theta] = cudaq::make_kernel<double>();
  // Allocate some qubits
  auto q = ansatz.qalloc(2);
  // Build up the circuit, use the acquired parameter
  ansatz.x(q[0]);
  ansatz.ry(theta, q[1]);
  ansatz.x<cudaq::ctrl>(q[1], q[0]);

  // Observe takes the kernel, the spin_op, and the concrete
  // parameters for the kernel.
  // This will be delegated to the remote simulator via HTTP requests.
  // Use `export CUDAQ_LOG_LEVEL=info` to inspect the server-client
  // communication.
  double energy = cudaq::observe(ansatz, h, .59);
  printf("Energy is %lf\n", energy);
  return 0;
}
