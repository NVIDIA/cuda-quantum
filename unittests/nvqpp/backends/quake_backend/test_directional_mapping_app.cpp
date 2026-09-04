/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include <cudaq.h>

#include <iostream>
#include <vector>

struct directional_mapping_cx_network {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector q(6);

    // Source SWAP and multi-control gates must be decomposed before mapping.
    // The SWAP acts while both qubits are zero, and the false q[1] control
    // keeps the CCX from changing q[2].
    swap(q[0], q[1]);
    x(q[0]);
    x<cudaq::ctrl>(q[0], q[1], q[2]);

    // Spread the initial one across nonlocal logical pairs.
    x<cudaq::ctrl>(q[0], q[5]);
    x<cudaq::ctrl>(q[5], q[2]);
    x<cudaq::ctrl>(q[2], q[4]);
    x<cudaq::ctrl>(q[4], q[1]);
    x<cudaq::ctrl>(q[1], q[3]);

    // Mix different control-target orders so routing must preserve direction.
    x<cudaq::ctrl>(q[3], q[0]);
    x<cudaq::ctrl>(q[2], q[5]);
    x<cudaq::ctrl>(q[4], q[2]);
    x<cudaq::ctrl>(q[1], q[4]);
    x<cudaq::ctrl>(q[5], q[1]);
    x<cudaq::ctrl>(q[0], q[3]);

    return cudaq::to_bools(mz(q));
  }
};

int main() {
  const std::vector<bool> expected{false, true, false, true, false, false};
  const auto results = cudaq::run(3, directional_mapping_cx_network{});
  if (results.size() != 3) {
    std::cerr << "Directional mapping returned an unexpected shot count.\n";
    return 1;
  }

  for (const auto &result : results) {
    if (result != expected) {
      std::cerr << "Directional mapping changed the logical CX result.\n";
      return 1;
    }
  }

  std::cout << "Directional CX mapping passed.\n";
  return 0;
}
