/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ %cpp_std --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t 
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>
#include <iostream>

struct bellKernel {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    cx(q[0], q[1]);
  }
};

__qpu__ void test_bell() {
  cudaq::qvector q(2);
  h(q[0]);
  cx(q[0], q[1]);
}

int main() {
  {
    auto state = cudaq::get_state(bellKernel{});
    std::vector<cudaq::complex> stateVec(4);
    if (state.is_on_gpu()) {
      state.to_host(stateVec.data(), stateVec.size());
      const std::vector<cudaq::complex> expected{M_SQRT1_2, 0., 0., M_SQRT1_2};
      const bool allClose = [&]() {
        for (std::size_t i = 0; i < 4; ++i)
          if (std::abs(expected[i] - stateVec[i]) > 1e-3)
            return false;
        return true;
      }();
      REMOTE_TEST_ASSERT(allClose);
    }
  }
  {
    auto state = cudaq::get_state(test_bell);
    std::vector<cudaq::complex> stateVec(4);
    if (state.is_on_gpu()) {
      state.to_host(stateVec.data(), stateVec.size());
      const std::vector<cudaq::complex> expected{M_SQRT1_2, 0., 0., M_SQRT1_2};
      const bool allClose = [&]() {
        for (std::size_t i = 0; i < 4; ++i)
          if (std::abs(expected[i] - stateVec[i]) > 1e-3)
            return false;
        return true;
      }();
      REMOTE_TEST_ASSERT(allClose);
    }
  }
  return 0;
}
