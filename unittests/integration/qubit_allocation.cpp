/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>

struct test_allocation {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    cudaq::qubit r;

    cnot(q, r);
    mz(q);
    mz(r);
  }
};

CUDAQ_TEST(AllocationTester, checkSimple) {
  test_allocation{}();

  auto counts = cudaq::sample(test_allocation{});
  EXPECT_EQ(2, counts.size());
  int c = 0;
  for (auto &[bits, count] : counts) {
    c += count;
    EXPECT_TRUE(bits == "00" || bits == "11");
  }
  EXPECT_EQ(c, 1000);
}
