/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecMpiCircuitSimulator.h"
#include "cudaq/ptsbe/PTSBESamplerImpl.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>

class MpSimTester
    : public cudaq::cusv::CuStateVecMpiCircuitSimulator<cudaq::real> {
  using Base = cudaq::cusv::CuStateVecCircuitSimulator<cudaq::real>;

public:
  using cudaq::cusv::CuStateVecMpiCircuitSimulator<
      cudaq::real>::sampleWithPTSBE;

  void advanceLocalRng(std::size_t count) {
    Base::generateRandomNumbers(count);
  }
};

namespace {

constexpr std::size_t numQubits = 6;

int minimumVisibleGpuCount() {
  int localCount = 0;
  if (cudaGetDeviceCount(&localCount) != cudaSuccess) {
    cudaGetLastError();
    localCount = 0;
  }
  const std::vector<int> local{localCount};
  std::vector<int> counts(cudaq::mpi::num_ranks());
  cudaq::mpi::all_gather(counts, local);
  return *std::min_element(counts.begin(), counts.end());
}

cudaq::kraus_channel makeHxChannel() {
  const double scale = std::sqrt(0.5);
  const double hadamardScale = 0.5;
  std::vector<cudaq::kraus_op> operators;
  operators.emplace_back(std::initializer_list<std::complex<cudaq::real>>{
      hadamardScale, hadamardScale, hadamardScale, -hadamardScale});
  operators.emplace_back(
      std::initializer_list<std::complex<cudaq::real>>{0.0, scale, scale, 0.0});
  return cudaq::kraus_channel(std::move(operators));
}

void runPtsbeOnDistributedState(bool desynchronizeRng) {
  MpSimTester simulator;
  std::vector<std::size_t> qubits(numQubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  cudaq::ExecutionContext context("ptsbe-sample", 2000);
  cudaq::detail::setExecutionContext(&context);
  simulator.configureExecutionContext(context);
  simulator.allocateQubits(numQubits);

  if (desynchronizeRng && cudaq::mpi::rank() == 1)
    simulator.advanceLocalRng(1);

  cudaq::ptsbe::PTSBatch batch;
  batch.trace.emplace_back(cudaq::ptsbe::TraceInstructionType::Noise, "hx",
                           std::vector<std::size_t>{0},
                           std::vector<std::size_t>{}, std::vector<double>{},
                           makeHxChannel());
  batch.measureQubits = {0};
  batch.trajectories.emplace_back(
      0, std::vector<cudaq::KrausSelection>{{0, {0}, "hx", 0, true}}, 0.5,
      1000);
  batch.trajectories.emplace_back(
      1, std::vector<cudaq::KrausSelection>{{0, {0}, "hx", 1, true}}, 0.5,
      1000);

  auto results = simulator.sampleWithPTSBE(batch);
  cudaq::detail::resetExecutionContext();
  ASSERT_EQ(results.size(), 2u);
  EXPECT_NEAR(results[0].probability("0"), 0.5, 0.1);
  EXPECT_NEAR(results[0].probability("1"), 0.5, 0.1);
  EXPECT_EQ(results[1].count("1"), 1000u);

  simulator.deallocateQubits(qubits);
}

} // namespace

class PtsbeMgpuTest : public ::testing::Test {
public:
  static void TearDownTestSuite() {
    cudaq::cusv::CuStateVecCommunicator::finalizeProvider();
  }
};

TEST_F(PtsbeMgpuTest, PtsbeDistributed) {
  if (minimumVisibleGpuCount() < 2)
    GTEST_SKIP() << "This regression requires two visible GPUs.";
  ASSERT_NE(std::getenv("CUDAQ_MGPU_NQUBITS_THRESH"), nullptr);
  runPtsbeOnDistributedState(false);
}

TEST_F(PtsbeMgpuTest, PtsbeDistributedDesync) {
  if (minimumVisibleGpuCount() < 2)
    GTEST_SKIP() << "This regression requires two visible GPUs.";
  ASSERT_NE(std::getenv("CUDAQ_MGPU_NQUBITS_THRESH"), nullptr);
  runPtsbeOnDistributedState(true);
}
