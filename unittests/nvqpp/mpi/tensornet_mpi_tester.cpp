/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <cudaq.h>
#include <gtest/gtest.h>

TEST(TensornetMPITester, checkInit) {
  EXPECT_TRUE(cudaq::mpi::is_initialized());
  std::cout << "Rank = " << cudaq::mpi::rank() << "\n";
}

TEST(TensornetMPITester, checkSimple) {
  constexpr std::size_t numQubits = 50;
  auto kernel = []() __qpu__ {
    cudaq::qvector q(numQubits);
    h(q[0]);
    for (int i = 0; i < numQubits - 1; i++)
      x<cudaq::ctrl>(q[i], q[i + 1]);
    mz(q);
  };

  auto counts = cudaq::sample(100, kernel);

  if (cudaq::mpi::rank() == 0) {
    EXPECT_EQ(2, counts.size());

    for (auto &[bits, count] : counts) {
      printf("Observed: %s, %lu\n", bits.data(), count);
      EXPECT_EQ(numQubits, bits.size());
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  cudaq::mpi::initialize();
  const auto testResult = RUN_ALL_TESTS();
  cudaq::mpi::finalize();
  return testResult;
}
