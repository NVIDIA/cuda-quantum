/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target density-matrix-cpu  %s -o %t && %t | FileCheck %s
// RUN: nvq++ --library-mode --target density-matrix-cpu  %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>

__qpu__ int test_kernel(int count) {
  unsigned result = 0;
  cudaq::qvector v(count);
  h(v[0]);
  for (int i = 0; i < count - 1; i++)
    cx(v[i], v[i + 1]);
  for (int i = 0; i < count; i++)
    if (mz(v[i]))
      result += 1;
  return result;
}

int main() {
  cudaq::depolarization_channel depol(0.5);
  cudaq::depolarization2 depol2(0.5);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::h>({0}, depol);
  noise.add_all_qubit_channel<cudaq::types::x>(depol2, /*numControls=*/1);
  {
    constexpr int numQubits = 4;
    auto results = cudaq::run(100, noise, test_kernel, numQubits);
    if (results.size() != 100) {
      printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
    } else {
      int idealCount = 0;
      int noisyCount = 0;
      int c = 0;
      for (auto i : results) {
        printf("%d: %d\n", c++, i);
        if (i == 0 || i == numQubits)
          idealCount++;
        else
          noisyCount++;
      }
      if ((idealCount + noisyCount) == 100) {
        if (noisyCount > 0)
          printf("success!\n");
        else
          printf("no noise effects can be observed. Something is wrong!\n");
      } else {
        printf("not enough count!\n");
      }
    }
  }

  // Async
  {
    constexpr int numQubits = 4;
    auto results =
        cudaq::run_async(/*qpu_id=*/0, 100, noise, test_kernel, numQubits)
            .get();
    if (results.size() != 100) {
      printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
    } else {
      int idealCount = 0;
      int noisyCount = 0;
      int c = 0;
      for (auto i : results) {
        printf("%d: %d\n", c++, i);
        if (i == 0 || i == numQubits)
          idealCount++;
        else
          noisyCount++;
      }
      if ((idealCount + noisyCount) == 100) {
        if (noisyCount > 0)
          printf("success!\n");
        else
          printf("no noise effects can be observed. Something is wrong!\n");
      } else {
        printf("not enough count!\n");
      }
    }
  }
  return 0;
}

// CHECK: success!
// CHECK: success!
