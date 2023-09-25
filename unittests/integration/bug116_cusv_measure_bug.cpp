/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"

CUDAQ_TEST(Bug116CuSVMeasure, checkMeasure) {
  struct run_circuit {
    auto operator()(bool mAll) __qpu__ {
      cudaq::qvector q(2);
      // Prep in state 10
      x(q[0]);
      // Measure the second qubit
      if (mAll)
        mz(q);
      else
        mz(q[1]);
    }
  };

  auto counts = cudaq::sample(10, run_circuit{}, true);
  EXPECT_EQ(1, counts.size());
  EXPECT_EQ("10", counts.begin()->first);
  EXPECT_EQ(10, counts.count("10"));

  auto counts2 = cudaq::sample(10, run_circuit{}, false);
  EXPECT_EQ(1, counts2.size());
  EXPECT_EQ("0", counts2.begin()->first);
  EXPECT_EQ(10, counts2.count("0"));
}
