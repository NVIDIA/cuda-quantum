/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"

#include <cudaq/algorithm.h>

inline std::vector<int> range(int N) {
  std::vector<int> vec(N);
  std::iota(vec.begin(), vec.end(), 0);
  return vec;
}

struct ghz {
  auto operator()(int N) __qpu__ {
    cudaq::qvector q(N);
    h(q[0]);
    for (auto i : range(N - 1)) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

CUDAQ_TEST(GHZSampleTester, checkSimple) {
  ghz{}(5);

  cudaq::set_random_seed(13);

  auto counts = cudaq::sample(ghz{}, 5);
  counts.dump();
  int counter = 0;
  for (auto &[bits, count] : counts) {
    counter += count;
    EXPECT_TRUE(bits == "00000" || bits == "11111");
  }
  EXPECT_EQ(counter, 1000);
  printf("Exp: %.16lf\n", counts.exp_val_z());
}

CUDAQ_TEST(GHZSampleTester, checkBroadcast) {

  cudaq::set_random_seed(13);

  std::vector<int> sizeVals(8);
  std::iota(sizeVals.begin(), sizeVals.end(), 3);
  {
    auto allCounts = cudaq::sample(ghz{}, cudaq::make_argset(sizeVals));

    std::cout << "allCounts size " << allCounts.size() << '\n';
    for (auto &counts : allCounts)
      counts.dump();

    int counter = 0;
    std::string first0 = "000", first1 = "111";
    for (auto &counts : allCounts) {
      for (auto &[bits, count] : counts) {
        counter += count;
        EXPECT_TRUE(bits == first0 || bits == first1);
      }
      EXPECT_EQ(counter, 1000);
      first0 += "0";
      first1 += "1";
      counter = 0;
    }
  }

  cudaq::set_random_seed(14);

  {
    auto allCounts = cudaq::sample(2000, ghz{}, cudaq::make_argset(sizeVals));

    std::cout << "allCounts size " << allCounts.size() << '\n';
    for (auto &counts : allCounts)
      counts.dump();

    int counter = 0;
    std::string first0 = "000", first1 = "111";
    for (auto &counts : allCounts) {
      for (auto &[bits, count] : counts) {
        counter += count;
        EXPECT_TRUE(bits == first0 || bits == first1);
      }
      EXPECT_EQ(counter, 2000);
      first0 += "0";
      first1 += "1";
      counter = 0;
    }
  }
}

CUDAQ_TEST(GHZSampleTester, checkBroadcastRepeatability) {
  std::vector<int> sizeVals(8);
  std::iota(sizeVals.begin(), sizeVals.end(), 3);

  cudaq::set_random_seed(13);
  auto allCounts1 = cudaq::sample(2000, ghz{}, cudaq::make_argset(sizeVals));

  cudaq::set_random_seed(13);
  auto allCounts2 = cudaq::sample(2000, ghz{}, cudaq::make_argset(sizeVals));

  cudaq::set_random_seed(14);
  auto allCounts3 = cudaq::sample(2000, ghz{}, cudaq::make_argset(sizeVals));

  EXPECT_EQ(allCounts1, allCounts2); // these should match
  EXPECT_NE(allCounts1, allCounts3); // these should NOT match
}