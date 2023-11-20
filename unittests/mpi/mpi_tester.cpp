/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <cudaq.h>
#include <gtest/gtest.h>
#include <random>

TEST(MPITester, checkInit) {
  EXPECT_TRUE(cudaq::mpi::is_initialized());
  std::cout << "Rank = " << cudaq::mpi::rank() << "\n";
}

TEST(MPITester, checkBroadcast) {
  constexpr std::size_t numElements = 100;
  const std::vector<double> expectedData =
      cudaq::random_vector(-M_PI, M_PI, numElements, /*seed = */ 1);
  // Only rank 0 has the data
  auto bcastVec = cudaq::mpi::rank() == 0
                      ? expectedData
                      : std::vector<double>(numElements, 0.0);
  // Broadcast
  cudaq::mpi::broadcast(bcastVec, 0);

  // All ranks have the same data
  for (std::size_t i = 0; i < bcastVec.size(); ++i) {
    EXPECT_EQ(bcastVec[i], expectedData[i])
        << "Broadcast data is corrupted at index " << i;
  }
}

TEST(MPITester, checkAllReduce) {
  {
    // Double type
    const std::vector<double> rankData = cudaq::random_vector(
        -M_PI, M_PI, cudaq::mpi::num_ranks(), /*seed = */ 1);
    const double localVal = rankData[cudaq::mpi::rank()];
    const double expectedSum = std::reduce(rankData.begin(), rankData.end());
    const double expectedProd = std::reduce(rankData.begin(), rankData.end(),
                                            1.0, std::multiplies<double>());
    const double mpiSumReduce =
        cudaq::mpi::all_reduce(localVal, std::plus<double>());
    const double mpiProdReduce =
        cudaq::mpi::all_reduce(localVal, std::multiplies<double>());
    EXPECT_NEAR(expectedSum, mpiSumReduce, 1e-12)
        << "All reduce SUM result does not match.";
    EXPECT_NEAR(expectedProd, mpiProdReduce, 1e-12)
        << "All reduce PROD result does not match.";
  }

  {
    // Float type
    const std::vector<double> rankData = cudaq::random_vector(
        -M_PI, M_PI, cudaq::mpi::num_ranks(), /*seed = */ 2);
    const double expectedSum = std::reduce(rankData.begin(), rankData.end());
    const double expectedProd = std::reduce(rankData.begin(), rankData.end(),
                                            1.0, std::multiplies<double>());
    const float localVal = rankData[cudaq::mpi::rank()];
    const float mpiSumReduce =
        cudaq::mpi::all_reduce(localVal, std::plus<float>());
    const float mpiProdReduce =
        cudaq::mpi::all_reduce(localVal, std::multiplies<float>());
    EXPECT_NEAR(expectedSum, mpiSumReduce, 1e-6)
        << "All reduce SUM result does not match.";
    EXPECT_NEAR(expectedProd, mpiProdReduce, 1e-6)
        << "All reduce PROD result does not match.";
  }
}

TEST(MPITester, checkAllGather) {
  constexpr std::size_t numElements = 10;
  const std::vector<double> expectedGatherData = cudaq::random_vector(
      -M_PI, M_PI, numElements * cudaq::mpi::num_ranks(), /*seed = */ 1);
  // Slice this vector to each rank
  const std::vector<double> rankData(
      expectedGatherData.begin() + numElements * cudaq::mpi::rank(),
      expectedGatherData.begin() + numElements * cudaq::mpi::rank() +
          numElements);
  EXPECT_EQ(rankData.size(), numElements);
  // Reconstruct the data vector with all_gather
  std::vector<double> gatherData(cudaq::mpi::num_ranks() * numElements);
  cudaq::mpi::all_gather(gatherData, rankData);
  for (std::size_t i = 0; i < gatherData.size(); ++i) {
    EXPECT_EQ(gatherData[i], expectedGatherData[i])
        << "AllGather data is corrupted at index " << i;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  cudaq::mpi::initialize();
  const auto testResult = RUN_ALL_TESTS();
  cudaq::mpi::finalize();
  return testResult;
}