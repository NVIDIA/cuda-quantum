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

#include <cudaq.h>

int main() {
  {
    auto kernel = cudaq::make_kernel();
    auto q = kernel.qalloc(2);
    kernel.h(q[0]);
    kernel.x<cudaq::ctrl>(q[0], q[1]);
    auto state = cudaq::get_state(kernel);
    assert(std::abs(M_SQRT1_2 - state[0].real()) < 1e-3);
    assert(std::abs(state[1].real()) < 1e-3);
    assert(std::abs(state[2].real()) < 1e-3);
    assert(std::abs(M_SQRT1_2 - state[3].real()) < 1e-3);
  }
// Skipped test due to a stability issue. See:
// https://github.com/NVIDIA/cuda-quantum/issues/1087
#if 0
  {
    auto &platform = cudaq::get_platform();

    auto num_qpus = platform.num_qpus();
    assert(num_qpus == 4);
    std::vector<cudaq::async_state_result> stateFutures;
    auto [kernel, num_qubits] = cudaq::make_kernel<int>();
    auto q = kernel.qalloc(num_qubits);
    kernel.h(q[0]);
    kernel.for_loop(0, num_qubits - 1, [&](auto &index) {
      kernel.x<cudaq::ctrl>(q[index], q[index + 1]);
    });
    for (std::size_t i = 0; i < num_qpus; ++i) {
      stateFutures.emplace_back(cudaq::get_state_async(i, kernel, i + 1));
    }
    for (std::size_t i = 0; i < num_qpus; ++i) {
      auto state = stateFutures[i].get();
      assert(state.get_shape()[0] == (1ULL << (i + 1)));
      assert(std::abs(M_SQRT1_2 - state[0].real()) < 1e-3);
      assert(std::abs(M_SQRT1_2 - state[(1ULL << (i + 1)) - 1].real()) < 1e-3);
    }
  }
#endif
  return 0;
}
