/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/distributed/mpi_plugin.h"
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
  {
    // double type
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
  {
    // int type
    const std::vector<int> rankData{cudaq::mpi::rank()};
    std::vector<int> expectedGatherData(cudaq::mpi::num_ranks());
    std::iota(expectedGatherData.begin(), expectedGatherData.end(),
              0); // Fill with 0, 1, ...
    std::vector<int> gatherData(cudaq::mpi::num_ranks());
    cudaq::mpi::all_gather(gatherData, rankData);
    for (std::size_t i = 0; i < gatherData.size(); ++i) {
      EXPECT_EQ(gatherData[i], expectedGatherData[i])
          << "AllGather data is corrupted at index " << i;
    }
  }
}

TEST(MPITester, checkAllGatherV) {
  const auto rank = cudaq::mpi::rank();
  const auto numRanks = cudaq::mpi::num_ranks();
  const int mySize = rank + 1;
  const int refSize = numRanks * (numRanks + 1) / 2;
  std::vector<int> sizes(numRanks);
  std::vector<int> offsets(numRanks);
  for (int iProc = 0; iProc < numRanks; iProc++) {
    sizes[iProc] = iProc + 1;
    offsets[iProc] = iProc * (iProc + 1) / 2;
  }
  const auto getSerialVector = [](int size, int offset) {
    std::vector<double> vector(size);
    for (int i = 0; i < size; ++i) {
      vector[i] = static_cast<double>(i + offset);
    }
    return vector;
  };
  const auto refVector = getSerialVector(refSize, 0);
  const int offset = offsets[rank];
  const auto myVector = getSerialVector(mySize, offset);
  std::vector<double> vector(refSize);
  auto *mpiPlugin = cudaq::mpi::getMpiPlugin();
  EXPECT_TRUE(mpiPlugin != nullptr);
  cudaqDistributedInterface_t *mpiInterface = mpiPlugin->get();
  EXPECT_TRUE(mpiInterface != nullptr);
  cudaqDistributedCommunicator_t *comm = mpiPlugin->getComm();
  EXPECT_TRUE(comm != nullptr);
  int initialized = 0;
  EXPECT_EQ(mpiInterface->initialized(&initialized), 0);
  EXPECT_EQ(initialized, 1);
  EXPECT_EQ(mpiInterface->AllgatherV(comm, myVector.data(), mySize,
                                     vector.data(), sizes.data(),
                                     offsets.data(), FLOAT_64),
            0);

  for (std::size_t i = 0; i < vector.size(); ++i) {
    EXPECT_EQ(vector[i], refVector[i])
        << "AllGatherV data is corrupted at index " << i;
  }
}

TEST(MPITester, checkSendAndRecv) {
  constexpr int nElems = 1;
  const auto rank = cudaq::mpi::rank();
  std::vector<double> sendBuffer(nElems);
  std::vector<double> recvBuffer(nElems);
  std::vector<double> refBuffer(nElems);
  const int sendRank = rank ^ 1;
  const int recvRank = sendRank;
  sendBuffer[0] = rank;
  refBuffer[0] = sendRank;
  auto *mpiPlugin = cudaq::mpi::getMpiPlugin();
  EXPECT_TRUE(mpiPlugin != nullptr);
  std::cout << "MPI plugin file: " << mpiPlugin->getPluginPath() << "\n";
  cudaqDistributedInterface_t *mpiInterface = mpiPlugin->get();
  EXPECT_TRUE(mpiInterface != nullptr);
  cudaqDistributedCommunicator_t *comm = mpiPlugin->getComm();
  EXPECT_TRUE(comm != nullptr);
  EXPECT_EQ(mpiInterface->RecvAsync(comm, recvBuffer.data(), nElems, FLOAT_64,
                                    recvRank, 0),
            0);
  EXPECT_EQ(mpiInterface->SendAsync(comm, sendBuffer.data(), nElems, FLOAT_64,
                                    sendRank, 0),
            0);
  EXPECT_EQ(mpiInterface->Synchronize(comm), 0);
  for (std::size_t i = 0; i < refBuffer.size(); ++i) {
    EXPECT_EQ(refBuffer[i], recvBuffer[i])
        << "Send-Receive data is corrupted at index " << i;
  }
}

TEST(MPITester, checkSendRecv) {
  constexpr int nElems = 1;
  const auto rank = cudaq::mpi::rank();
  std::vector<double> sendBuffer(nElems);
  std::vector<double> recvBuffer(nElems);
  std::vector<double> refBuffer(nElems);
  const int sendRecvRank = rank ^ 1;
  sendBuffer[0] = rank;
  refBuffer[0] = sendRecvRank;
  auto *mpiPlugin = cudaq::mpi::getMpiPlugin();
  EXPECT_TRUE(mpiPlugin != nullptr);
  cudaqDistributedInterface_t *mpiInterface = mpiPlugin->get();
  EXPECT_TRUE(mpiInterface != nullptr);
  cudaqDistributedCommunicator_t *comm = mpiPlugin->getComm();
  EXPECT_TRUE(comm != nullptr);
  EXPECT_EQ(mpiInterface->SendRecvAsync(comm, sendBuffer.data(),
                                        recvBuffer.data(), nElems, FLOAT_64,
                                        sendRecvRank, 0),
            0);
  EXPECT_EQ(mpiInterface->Synchronize(comm), 0);
  for (std::size_t i = 0; i < refBuffer.size(); ++i) {
    EXPECT_EQ(refBuffer[i], recvBuffer[i])
        << "Send-Receive data is corrupted at index " << i;
  }
}

TEST(MPITester, checkCommDup) {
  auto *mpiPlugin = cudaq::mpi::getMpiPlugin();
  EXPECT_TRUE(mpiPlugin != nullptr);
  const int origSize = cudaq::mpi::num_ranks();
  cudaqDistributedInterface_t *mpiInterface = mpiPlugin->get();
  EXPECT_TRUE(mpiInterface != nullptr);
  cudaqDistributedCommunicator_t *comm = mpiPlugin->getComm();
  EXPECT_TRUE(comm != nullptr);
  int initialized = 0;
  EXPECT_EQ(mpiInterface->initialized(&initialized), 0);
  EXPECT_EQ(initialized, 1);
  cudaqDistributedCommunicator_t *dupComm = nullptr;
  EXPECT_EQ(mpiInterface->CommDup(comm, &dupComm), 0);
  EXPECT_TRUE(dupComm != nullptr);
  int size = 0;
  EXPECT_EQ(mpiInterface->getNumRanks(dupComm, &size), 0);
  EXPECT_GT(size, 0);
  EXPECT_EQ(size, origSize);
}

TEST(MPITester, checkCommSplit) {
  auto *mpiPlugin = cudaq::mpi::getMpiPlugin();
  EXPECT_TRUE(mpiPlugin != nullptr);
  cudaqDistributedInterface_t *mpiInterface = mpiPlugin->get();
  EXPECT_TRUE(mpiInterface != nullptr);
  cudaqDistributedCommunicator_t *comm = mpiPlugin->getComm();
  EXPECT_TRUE(comm != nullptr);
  int initialized = 0;
  EXPECT_EQ(mpiInterface->initialized(&initialized), 0);
  EXPECT_EQ(initialized, 1);
  cudaqDistributedCommunicator_t *dupComm = nullptr;
  EXPECT_EQ(mpiInterface->CommSplit(comm, /*color=*/cudaq::mpi::rank(),
                                    /*key=*/0, &dupComm),
            0);
  EXPECT_TRUE(dupComm != nullptr);
  int size = 0;
  EXPECT_EQ(mpiInterface->getNumRanks(dupComm, &size), 0);
  EXPECT_EQ(size, 1);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  cudaq::mpi::initialize();
  const auto testResult = RUN_ALL_TESTS();
  cudaq::mpi::finalize();
  return testResult;
}
