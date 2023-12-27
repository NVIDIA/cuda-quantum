/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "common/FmtCore.h"
#include "cudaq/utils/cudaq_utils.h"

TEST(UtilsTester, checkRange) {
  {
    auto v = cudaq::range(0, 10, 2);
    std::cout << fmt::format("{}", fmt::join(v, ",")) << "\n";
    EXPECT_EQ(5, v.size());
    std::vector<int> expected{0, 2, 4, 6, 8};
    EXPECT_EQ(expected, v);
  }

  {
    auto v = cudaq::range(10);
    std::cout << fmt::format("{}", fmt::join(v, ",")) << "\n";
    EXPECT_EQ(10, v.size());
    std::vector<int> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(expected, v);
  }

  {
    auto v = cudaq::range(10, 2, -1);
    std::cout << fmt::format("{}", fmt::join(v, ",")) << "\n";
    EXPECT_EQ(8, v.size());
    std::vector<int> expected{10, 9, 8, 7, 6, 5, 4, 3};
    EXPECT_EQ(expected, v);
  }
}
