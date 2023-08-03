/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq.h"
#include "cudaq/photonics.h"

TEST(PhotonicsTester, checkSimple) {

  struct test {
    auto operator()() __qpu__ {
      cudaq::qreg<cudaq::dyn, 3> qutrits(2);
      plusGate(qutrits[0]);
      plusGate(qutrits[1]);
      plusGate(qutrits[1]);
      return mz(qutrits);
    }
  };

  struct test2 {
    void operator()() __qpu__ {
      cudaq::qreg<cudaq::dyn, 3> qutrits(2);
      plusGate(qutrits[0]);
      plusGate(qutrits[1]);
      plusGate(qutrits[1]);
      mz(qutrits);
    }
  };

  auto res = test{}();
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], 2);

  cudaq::sample(test2{});
}
