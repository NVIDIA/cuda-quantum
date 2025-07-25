/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std %s -o %t && %t
// RUN: nvq++ %cpp_std --target quantinuum --emulate -fenable-cudaq-run -%s -o %t && CUDAQ_ENABLE_QUANTUM_DEVICE_RUN=1 %t
// clang-format on

#include <cstdio>
#include <cudaq.h>
#include <cudaq/algorithm.h>

__qpu__ uint64_t copy_to_integer(const std::vector<int> applyX) {
  cudaq::qvector q(applyX.size());

  for (std::size_t i = 0; i < applyX.size(); i++) {
    if (applyX[i]) {
      x(q[i]);
    }
  }

  auto results = mz(q);

  std::vector<cudaq::measure_result> copy(applyX.size());
  int i = 0;
  for (auto s : results) {
    copy[i++] = s;
  }

  return cudaq::to_integer(copy);
}

int main() {
  std::vector<int> test{1, 1, 1, 1};

  auto r = cudaq::run(1, copy_to_integer, test);
  if (r[0] != 15) {
    printf("1111 has to map to 15.\n");
    return 1;
  }

  return 0;
}
