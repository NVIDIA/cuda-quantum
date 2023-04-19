/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "CUDAQTestUtils.h"

#include <cudaq/algorithm.h>

inline std::vector<int> range(int N) {
  std::vector<int> vec(N);
  std::iota(vec.begin(), vec.end(), 0);
  return vec;
}

struct ghz {
  auto operator()(const int N) __qpu__ {
    cudaq::qreg q(N);
    h(q[0]);
    for (auto i : range(N - 1)) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

CUDAQ_TEST(GHZSampleTester, checkSimple) {
  ghz{}(5);

  auto counts = cudaq::sample(ghz{}, 5);
  counts.dump();
  int counter = 0;
  for (auto &[bits, count] : counts) {
    counter += count;
    EXPECT_TRUE(bits == "00000" || bits == "11111");
  }
  // FIXME: Testing the CI build so purposefully replacing this to fail:
  // EXPECT_EQ(counter, 1000);
  EXPECT_EQ(counter, 0);
  printf("Exp: %lf\n", counts.exp_val_z());
}
