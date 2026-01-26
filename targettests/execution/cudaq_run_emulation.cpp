/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/


// clang-format off
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t |& FileCheck %s -check-prefix=FAIL
// RUN: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t | FileCheck %s; fi
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

__qpu__ std::vector<bool> mz_test(int count) {
  cudaq::qvector v(count);
  h(v);
  return mz(v);
}

int main() {
  size_t shots = 20;
  int c = 0;
  {
    constexpr int numQubits = 4;
    auto results = cudaq::run(shots, test_kernel, numQubits);
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results) {
        printf("%d: %d\n", c++, i);
        if (i != 0 && i != 4)
          break;
      }
      if (c == shots)
        printf("success!\n");
    }
  }

  // Also test asynchronous API
  {
    const auto results =
        cudaq::run_async(/*qpu_id=*/0, shots, mz_test, 2).get();
    c = 0;
    if (results.size() != shots) {
      printf("FAILED! Expected %lu shots. Got %lu\n", shots, results.size());
    } else {
      for (auto i : results) {
        printf("%d: %d %d\n", c++, (bool)i[0], (bool)i[1]);
      }
      if (c == shots)
        printf("success async!\n");
    }
  }

  return 0;
}

// FAIL: `run` is not yet supported on this target
// CHECK: success!
// CHECK: success async!
