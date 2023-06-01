/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"

CUDAQ_TEST(NegativeControlsTester, checkSimple) {

  auto kernel = []() __qpu__ {
    cudaq::qarray<2> q;
    x<cudaq::ctrl>(!q[0], q[1]);
    mz(q);
  };

  auto counts = cudaq::sample(kernel);
  counts.dump();

  int counter = 0;
  for (auto &[bits, count] : counts) {
    counter += count;
    EXPECT_TRUE(bits == "01");
  }

  EXPECT_EQ(counter, 1000);

  auto kernel2 = []() __qpu__ {
    cudaq::qarray<4> q;
    x<cudaq::ctrl>(!q[0], !q[1], !q[2], q[3]);
    mz(q);
  };

  counts = cudaq::sample(kernel2);
  counts.dump();

  counter = 0;
  for (auto &[bits, count] : counts) {
    counter += count;
    EXPECT_TRUE(bits == "0001");
  }

  EXPECT_EQ(counter, 1000);

  auto kernel3 = []() __qpu__ {
    cudaq::qarray<2> q;
    x(q.front(2));
    x<cudaq::ctrl>(!q[0], q[1]);
    mz(q);
  };

  counts = cudaq::sample(kernel3);
  counts.dump();

  counter = 0;
  for (auto &[bits, count] : counts) {
    counter += count;
    EXPECT_TRUE(bits == "11");
  }

  EXPECT_EQ(counter, 1000);
}
