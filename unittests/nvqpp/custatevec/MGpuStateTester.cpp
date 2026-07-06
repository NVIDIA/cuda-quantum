/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecMpiCircuitSimulator.h"
#include "nvqir/Gates.h"
#include "cudaq/distributed/mpi_plugin.h"
#include <execution>
#include <gtest/gtest.h>
#include <mpi.h>
#include <type_traits>

namespace {
double randomAngle() {
  // Note: all processes should be using the same seed
  static std::mt19937 engine(1);
  static std::uniform_real_distribution<> dist(-M_PI, M_PI);
  return dist(engine);
}

class Simulator
    : public cudaq::cusv::CuStateVecMpiCircuitSimulator<cudaq::real> {
public:
  using cudaq::cusv::CuStateVecMpiCircuitSimulator<cudaq::real>::flushGateQueue;
  using cudaq::cusv::CuStateVecMpiCircuitSimulator<
      cudaq::real>::getSimulationState;

  bool importDeviceState(const void *data, std::size_t size) {
    return state().setStateFromDevicePointer(data, size);
  }

  std::vector<double> generateDistributedRandomNumbers(std::size_t count) {
    return generateRandomNumbers(count);
  }

  std::vector<double> generateLocalRandomNumbers(std::size_t count) {
    return cudaq::cusv::CuStateVecCircuitSimulator<
        cudaq::real>::generateRandomNumbers(count);
  }
};

class NoHostTransferState
    : public cudaq::cusv::CuStateVecSimulationState<cudaq::real> {
public:
  using CuStateVecSimulationState::CuStateVecSimulationState;

  void toHost(std::complex<double> *, std::size_t) const override {
    throw std::runtime_error("unexpected FP64 host transfer");
  }
  void toHost(std::complex<float> *, std::size_t) const override {
    throw std::runtime_error("unexpected FP32 host transfer");
  }
};

Simulator &getSimulator() {
  static Simulator sim;
  return sim;
}

int allreduceCalls = 0;
int allreduceInPlaceCalls = 0;

int spyAllreduce(const cudaqDistributedCommunicator_t *, const void *, void *,
                 int32_t, DataType, ReduceOp) {
  ++allreduceCalls;
  return 0;
}

int spyAllreduceInPlace(const cudaqDistributedCommunicator_t *, void *, int32_t,
                        DataType, ReduceOp) {
  ++allreduceInPlaceCalls;
  return 0;
}

constexpr auto simulationPrecision = std::is_same_v<cudaq::real, float>
                                         ? cudaq::simulation_precision::fp32
                                         : cudaq::simulation_precision::fp64;
constexpr double tolerance = std::is_same_v<cudaq::real, float> ? 1e-6 : 1e-9;

int minimumVisibleGpuCount() {
  int32_t localCount = 0;
  if (cudaGetDeviceCount(&localCount) != cudaSuccess) {
    cudaGetLastError();
    localCount = 0;
  }
  int32_t minimumCount = 0;
  if (MPI_Allreduce(&localCount, &minimumCount, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
    return 0;
  return minimumCount;
}

// Number of qubits to run these tests
constexpr std::size_t numQubits = 15;
} // namespace

TEST(MGpuTesterMultiProcesses, BasicCheck) {
  // This must be set before running this test to make sure the state is
  // distributed.
  EXPECT_TRUE(std::getenv("CUDAQ_MGPU_NQUBITS_THRESH") != nullptr);
  const auto nQubitsMGPUThreshold =
      std::atoi(std::getenv("CUDAQ_MGPU_NQUBITS_THRESH"));
  // The threshold must be set to a small value.
  EXPECT_LE(nQubitsMGPUThreshold, 10);
}

TEST(MGpuTesterMultiProcesses, BelowThresholdStateSupportsNonPowerOfTwoRanks) {
  auto &sim = getSimulator();
  if (cudaq::mpi::num_ranks() != 6)
    GTEST_SKIP() << "This regression requires six MPI ranks.";
  const auto threshold = static_cast<std::size_t>(
      std::atoi(std::getenv("CUDAQ_MGPU_NQUBITS_THRESH")));
  ASSERT_GT(threshold, std::size_t{1});
  const auto qubitCount = threshold - 1;
  sim.allocateQubits(qubitCount);
  sim.h(0);
  sim.synchronize();
  auto state = sim.getSimulationState();
  auto *const exState =
      dynamic_cast<cudaq::cusv::CuStateVecSimulationState<cudaq::real> *>(
          state.get());
  ASSERT_NE(exState, nullptr);
  EXPECT_EQ(exState->state().distributionType(),
            CUSTATEVEC_EX_SV_DISTRIBUTION_SINGLE_DEVICE);
  EXPECT_EQ(exState->state().numWires(), static_cast<int32_t>(qubitCount));
  EXPECT_NEAR(std::abs(state->getAmplitude(std::vector<int>(qubitCount, 0))),
              M_SQRT1_2, tolerance);
  std::vector<int> one(qubitCount, 0);
  one.front() = 1;
  EXPECT_NEAR(std::abs(state->getAmplitude(one)), M_SQRT1_2, tolerance);
  std::vector<std::size_t> qubits(qubitCount);
  std::iota(qubits.begin(), qubits.end(), 0);
  sim.deallocateQubits(qubits);
}

TEST(MGpuTesterMultiProcesses, GrowsSmallReplicatedStateAcrossThreshold) {
  if (minimumVisibleGpuCount() < 2)
    GTEST_SKIP() << "This regression requires two visible GPUs.";
  auto &sim = getSimulator();
  if (cudaq::mpi::num_ranks() != 4)
    GTEST_SKIP() << "This regression requires four MPI ranks.";

  const std::vector<cudaq::complex> plusState{cudaq::complex{M_SQRT1_2, 0.0},
                                              cudaq::complex{M_SQRT1_2, 0.0}};
  sim.allocateQubits(1, plusState.data(), simulationPrecision);
  sim.allocateQubits(1);
  sim.synchronize();

  auto state = sim.getSimulationState();
  auto *const exState =
      dynamic_cast<cudaq::cusv::CuStateVecSimulationState<cudaq::real> *>(
          state.get());
  ASSERT_NE(exState, nullptr);
  EXPECT_EQ(exState->state().distributionType(),
            CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS);
  EXPECT_NEAR(std::abs(state->getAmplitude({0, 0})), M_SQRT1_2, tolerance);
  EXPECT_NEAR(std::abs(state->getAmplitude({1, 0})), M_SQRT1_2, tolerance);
  EXPECT_NEAR(std::abs(state->getAmplitude({0, 1})), 0.0, tolerance);
  EXPECT_NEAR(std::abs(state->getAmplitude({1, 1})), 0.0, tolerance);
  sim.deallocateQubits({0, 1});
}

TEST(MGpuTesterMultiProcesses, ValidatesMigrationLevelWithoutMigrationWires) {
  if (minimumVisibleGpuCount() < 2)
    GTEST_SKIP() << "This regression requires two visible GPUs.";
  auto &sim = getSimulator();
  if (!std::getenv("CUDAQ_HOST_DEVICE_MIGRATION_LEVEL"))
    GTEST_SKIP() << "This regression requires an invalid migration level.";
  const auto threshold = static_cast<std::size_t>(
      std::atoi(std::getenv("CUDAQ_MGPU_NQUBITS_THRESH")));
  EXPECT_THROW(sim.allocateQubits(threshold), std::invalid_argument);
}

TEST(MGpuTesterMultiProcesses, GetSimulationStateRejectsRepeatedCall) {
  auto &sim = getSimulator();
  sim.allocateQubits(1);
  auto state = sim.getSimulationState();
  ASSERT_NE(state, nullptr);
  EXPECT_THROW(sim.getSimulationState(), std::runtime_error);
  sim.deallocateQubits({0});
}

TEST(MGpuTesterMultiProcesses, DistributedGpuRandomAdvancesEveryRank) {
  constexpr std::size_t count = 100001;
  auto &sim = getSimulator();
  const auto first = sim.generateDistributedRandomNumbers(count);
  const auto second = sim.generateDistributedRandomNumbers(count);
  const auto local = sim.generateLocalRandomNumbers(count);
  ASSERT_EQ(first.size(), count);
  ASSERT_EQ(second.size(), count);
  ASSERT_EQ(local.size(), count);

  std::vector<double> minimum(count);
  std::vector<double> maximum(count);
  ASSERT_EQ(MPI_Allreduce(local.data(), minimum.data(), static_cast<int>(count),
                          MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD),
            MPI_SUCCESS);
  ASSERT_EQ(MPI_Allreduce(local.data(), maximum.data(), static_cast<int>(count),
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
            MPI_SUCCESS);
  for (std::size_t index = 0; index < count; ++index)
    EXPECT_NEAR(minimum[index], maximum[index], 1e-12);
}

TEST(MGpuTesterMultiProcesses, GetAmplitude) {
  const double theta = randomAngle();
  const double phi = randomAngle();
  const double lambda = randomAngle();
  const auto u3Mat = nvqir::getGateByName(
      nvqir::GateName::U3, std::vector<double>{theta, phi, lambda});
  {
    auto &sim = getSimulator();
    sim.allocateQubits(numQubits);
    sim.u3(theta, phi, lambda, 0);
    for (std::size_t i = 0; i < numQubits - 1; ++i)
      sim.x({i}, i + 1);
    sim.flushGateQueue();
    auto state = sim.getSimulationState();
    auto *const exState =
        dynamic_cast<cudaq::cusv::CuStateVecSimulationState<cudaq::real> *>(
            state.get());
    ASSERT_NE(exState, nullptr);
    EXPECT_EQ(exState->state().distributionType(),
              CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS);
    EXPECT_TRUE(state->isDeviceData());
    EXPECT_EQ(state->getNumElements(), std::size_t{1} << numQubits);
    // state->dump(std::cout);
    // These 2 basis state should be living on different ranks
    EXPECT_NEAR(std::abs(state->getAmplitude(std::vector<int>(numQubits, 0)) -
                         u3Mat[0]),
                0.0, tolerance);
    EXPECT_NEAR(std::abs(state->getAmplitude(std::vector<int>(numQubits, 1)) -
                         u3Mat[2]),
                0.0, tolerance);
    std::vector<std::size_t> qubitIdxs(numQubits);
    std::iota(qubitIdxs.begin(), qubitIdxs.end(), 0);
    sim.deallocateQubits(qubitIdxs);
  }
}

TEST(MGpuTesterMultiProcesses, ToHostBroadCast) {
  const double theta = randomAngle();
  const double phi = randomAngle();
  const double lambda = randomAngle();
  const auto u3Mat = nvqir::getGateByName(
      nvqir::GateName::U3, std::vector<double>{theta, phi, lambda});
  auto &sim = getSimulator();
  sim.allocateQubits(numQubits);
  sim.u3(theta, phi, lambda, 0);
  for (std::size_t i = 0; i < numQubits - 1; ++i)
    sim.x({i}, i + 1);
  sim.flushGateQueue();
  auto state = sim.getSimulationState();
  std::vector<cudaq::complex> stateVec(1 << numQubits);
  // state->dump(std::cout);
  state->toHost(stateVec.data(), stateVec.size());
  // This check will be run on all ranks, i.e., all ranks now have a global view
  // of the state vector
  for (std::size_t i = 0; i < stateVec.size(); ++i) {
    if (i == 0) {
      EXPECT_NEAR(std::abs(stateVec[i] - static_cast<cudaq::complex>(u3Mat[0])),
                  0.0, tolerance);
    } else if (i == stateVec.size() - 1) {
      EXPECT_NEAR(std::abs(stateVec[i] - static_cast<cudaq::complex>(u3Mat[2])),
                  0.0, tolerance);
    } else {
      // All others are zeros
      EXPECT_NEAR(std::abs(stateVec[i]), 0.0, tolerance);
    }
  }
  std::vector<std::size_t> qubitIdxs(numQubits);
  std::iota(qubitIdxs.begin(), qubitIdxs.end(), 0);
  sim.deallocateQubits(qubitIdxs);
}

TEST(MGpuTesterMultiProcesses, ToHostRankLocal) {
  const double theta = randomAngle();
  const double phi = randomAngle();
  const double lambda = randomAngle();
  const auto u3Mat = nvqir::getGateByName(
      nvqir::GateName::U3, std::vector<double>{theta, phi, lambda});
  auto &sim = getSimulator();
  sim.allocateQubits(numQubits);
  sim.u3(theta, phi, lambda, 0);
  for (std::size_t i = 0; i < numQubits - 1; ++i)
    sim.x({i}, i + 1);
  sim.flushGateQueue();
  auto state = sim.getSimulationState();
  // Get a sub-state vector view (each rank sees different data)
  const auto numProcs = cudaq::mpi::num_ranks();
  std::vector<cudaq::complex> subStateVec((1 << numQubits) / numProcs);
  state->toHost(subStateVec.data(), subStateVec.size());
  // Each rank only have a local view about the state
  const auto rank = cudaq::mpi::rank();
  for (std::size_t i = 0; i < subStateVec.size(); ++i) {
    if (i == 0 && rank == 0) {
      // |000..00> lives on rank 0
      EXPECT_NEAR(
          std::abs(subStateVec[i] - static_cast<cudaq::complex>(u3Mat[0])), 0.0,
          tolerance);
    } else if (i == subStateVec.size() - 1 && rank == numProcs - 1) {
      // |111..11> lives on the last rank
      EXPECT_NEAR(
          std::abs(subStateVec[i] - static_cast<cudaq::complex>(u3Mat[2])), 0.0,
          tolerance);
    } else {
      // All others are zeros
      EXPECT_NEAR(std::abs(subStateVec[i]), 0.0, tolerance);
    }
  }
  std::vector<std::size_t> qubitIdxs(numQubits);
  std::iota(qubitIdxs.begin(), qubitIdxs.end(), 0);
  sim.deallocateQubits(qubitIdxs);
}

TEST(MGpuTesterMultiProcesses, Overlap) {
  auto &sim = getSimulator();
  sim.allocateQubits(numQubits);
  for (std::size_t i = 0; i < numQubits; ++i)
    sim.u3(randomAngle(), randomAngle(), randomAngle(), i);
  sim.flushGateQueue();
  auto state1 = sim.getSimulationState();
  std::vector<std::size_t> qubitIdxs(numQubits);
  std::iota(qubitIdxs.begin(), qubitIdxs.end(), 0);
  std::vector<cudaq::complex> stateVec1(1 << numQubits);
  state1->toHost(stateVec1.data(), stateVec1.size());
  sim.deallocateQubits(qubitIdxs);

  sim.allocateQubits(numQubits);
  for (std::size_t i = 0; i < numQubits; ++i)
    sim.u3(randomAngle(), randomAngle(), randomAngle(), i);
  sim.flushGateQueue();
  auto state2 = sim.getSimulationState();
  std::vector<cudaq::complex> stateVec2(1 << numQubits);
  state2->toHost(stateVec2.data(), stateVec2.size());
  sim.deallocateQubits(qubitIdxs);

  // GPU overlap calculation (multi-GPU)
  const auto overlap = state1->overlap(*state2);
  std::cout << "Overlap = " << overlap << "\n";

  const auto overlapCheck = [&]() {
    std::complex<double> sum = 0;
    for (std::size_t i = 0; i < (1 << numQubits); ++i)
      sum += std::conj(stateVec1[i]) * stateVec2[i];
    return std::abs(sum);
  }();
  std::cout << "Check = " << overlapCheck << "\n";
  EXPECT_NEAR(std::abs(overlapCheck - overlap), 0.0, tolerance);
}

TEST(MGpuTesterMultiProcesses, AddQubitsInState) {
  auto &sim = getSimulator();
  std::vector<cudaq::complex> catState(1 << numQubits, 0.0);
  catState.front() = M_SQRT1_2;
  catState.back() = M_SQRT1_2;
  sim.allocateQubits(numQubits, catState.data(), simulationPrecision);

  sim.flushGateQueue();
  auto state = sim.getSimulationState();
  const std::vector<int> allZeroBasisState(numQubits, 0);
  const std::vector<int> allOneBasisState(numQubits, 1);
  EXPECT_NEAR(std::abs(state->getAmplitude(allZeroBasisState)), M_SQRT1_2,
              tolerance);
  EXPECT_NEAR(std::abs(state->getAmplitude(allZeroBasisState)), M_SQRT1_2,
              tolerance);
  std::vector<std::size_t> qubitIdxs(numQubits);
  std::iota(qubitIdxs.begin(), qubitIdxs.end(), 0);
  sim.deallocateQubits(qubitIdxs);
}

TEST(MGpuTesterMultiProcesses, ImportsOnlyAssignedDeviceSubStates) {
  auto &sim = getSimulator();
  std::vector<cudaq::complex> expected(std::size_t{1} << numQubits, 0.0);
  expected.front() = M_SQRT1_2;
  expected.back() = M_SQRT1_2;
  cudaq::complex *deviceState = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&deviceState),
                                    expected.size() * sizeof(cudaq::complex)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(deviceState, expected.data(),
                                    expected.size() * sizeof(cudaq::complex),
                                    cudaMemcpyHostToDevice));

  sim.allocateQubits(numQubits);
  sim.flushGateQueue();
  EXPECT_TRUE(sim.importDeviceState(deviceState, expected.size()));
  auto state = sim.getSimulationState();
  auto *const exState =
      dynamic_cast<cudaq::cusv::CuStateVecSimulationState<cudaq::real> *>(
          state.get());
  ASSERT_NE(exState, nullptr);
  EXPECT_NEAR(std::abs(state->getAmplitude(std::vector<int>(numQubits, 0))),
              M_SQRT1_2, tolerance);
  EXPECT_NEAR(std::abs(state->getAmplitude(std::vector<int>(numQubits, 1))),
              M_SQRT1_2, tolerance);

  std::vector<std::size_t> qubits(numQubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  sim.deallocateQubits(qubits);
  EXPECT_EQ(cudaSuccess, cudaFree(deviceState));
}

TEST(MGpuTesterMultiProcesses, AddQubitsInStateGetState) {
  auto &sim = getSimulator();

  sim.allocateQubits(numQubits);
  sim.h(0);
  for (int q = 0; q < numQubits - 1; ++q)
    sim.x({static_cast<std::size_t>(q)}, q + 1);
  sim.flushGateQueue();
  auto state = sim.getSimulationState();
  std::vector<std::size_t> qubitIdxs(numQubits);
  std::iota(qubitIdxs.begin(), qubitIdxs.end(), 0);
  sim.deallocateQubits(qubitIdxs);

  sim.allocateQubits(numQubits, state.get());
  for (int q = 0; q < numQubits; ++q)
    sim.x(q);
  sim.flushGateQueue();
  auto state2 = sim.getSimulationState();
  sim.deallocateQubits(qubitIdxs);
  const std::vector<int> allZeroBasisState(numQubits, 0);
  const std::vector<int> allOneBasisState(numQubits, 1);
  EXPECT_NEAR(std::abs(state2->getAmplitude(allZeroBasisState)), M_SQRT1_2,
              tolerance);
  EXPECT_NEAR(std::abs(state2->getAmplitude(allOneBasisState)), M_SQRT1_2,
              tolerance);
}

TEST(MGpuTesterMultiProcesses, ResizeAddQubitsInState) {
  auto &sim = getSimulator();
  // Prepare GHZ state: (distributed) |00...> + |11...>
  sim.allocateQubits(numQubits);
  sim.h(0);
  for (int q = 0; q < numQubits - 1; ++q)
    sim.x({static_cast<std::size_t>(q)}, q + 1);
  sim.flushGateQueue();

  // |111> state
  std::vector<cudaq::complex> oneState(1 << 3, 0.0);
  oneState.back() = 1.0;
  sim.allocateQubits(3, oneState.data(), simulationPrecision);
  sim.flushGateQueue();
  auto state = sim.getSimulationState();
  std::vector<int> allZeroBasisState(numQubits, 0);
  std::vector<int> allOneBasisState(numQubits, 1);
  for (std::size_t i = 0; i < 3; ++i) {
    allZeroBasisState.emplace_back(1);
    allOneBasisState.emplace_back(1);
  }
  EXPECT_NEAR(std::abs(state->getAmplitude(allZeroBasisState)), M_SQRT1_2,
              tolerance);
  EXPECT_NEAR(std::abs(state->getAmplitude(allZeroBasisState)), M_SQRT1_2,
              tolerance);
  std::vector<std::size_t> qubitIdxs(numQubits + 3);
  std::iota(qubitIdxs.begin(), qubitIdxs.end(), 0);
  sim.deallocateQubits(qubitIdxs);
}

template <typename T>
std::vector<T> kronProd(const std::vector<T> &a, const std::vector<T> &b) {
  std::vector<T> result(a.size() * b.size());
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = 0; j < b.size(); ++j)
      result[i * b.size() + j] = a[i] * b[j];

  return result;
}

static std::vector<cudaq::complex> randomState(int numQubits) {
  std::vector<cudaq::complex> stateVec(1ULL << numQubits);
  std::generate(stateVec.begin(), stateVec.end(), []() -> cudaq::complex {
    thread_local std::default_random_engine
        generator; // thread_local so we don't have to do any locking
    thread_local std::normal_distribution<double> distribution(
        0.0, 1.0); // mean = 0.0, stddev = 1.0
    return {static_cast<cudaq::real>(distribution(generator)),
            static_cast<cudaq::real>(distribution(generator))};
  });

  const double norm =
      std::sqrt(std::accumulate(stateVec.begin(), stateVec.end(), 0.0,
                                [](double accumulatedNorm, cudaq::complex val) {
                                  return accumulatedNorm + std::norm(val);
                                }));
  std::transform(
      stateVec.begin(), stateVec.end(), stateVec.begin(),
      [norm](cudaq::complex x) { return x / static_cast<cudaq::real>(norm); });
  return stateVec;
}

TEST(MGpuTesterMultiProcesses, ResizeAddQubitsInStateVecGeneral) {
  auto &sim = getSimulator();

  std::vector<cudaq::complex> state1 = randomState(numQubits);
  sim.allocateQubits(numQubits, state1.data(), simulationPrecision);
  std::vector<cudaq::complex> state2 = randomState(3);
  sim.allocateQubits(3, state2.data(), simulationPrecision);

  sim.synchronize();
  auto state = sim.getSimulationState();
  auto expectedState = kronProd(state2, state1);
  std::vector<cudaq::complex> resultStateHost(expectedState.size());
  state->toHost(resultStateHost.data(), resultStateHost.size());
  for (std::size_t i = 0; i < resultStateHost.size(); ++i)
    EXPECT_NEAR(std::abs(resultStateHost[i] - expectedState[i]), 0.0,
                tolerance);
  std::vector<std::size_t> qubitIdxs(numQubits + 3);
  std::iota(qubitIdxs.begin(), qubitIdxs.end(), 0);
  sim.deallocateQubits(qubitIdxs);
}

TEST(MGpuTesterMultiProcesses, ResizeAddQubitsInStateGeneral) {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, M_PI);
  auto &sim = getSimulator();
  constexpr int numAddedQubits = 3;
  sim.allocateQubits(numAddedQubits);
  // Build some random state via circuit
  for (std::size_t i = 0; i < numAddedQubits; ++i)
    sim.rx(distribution(generator), i);
  sim.flushGateQueue();
  auto randState = sim.getSimulationState();
  {
    std::vector<std::size_t> qubitIdxs(numAddedQubits);
    std::iota(qubitIdxs.begin(), qubitIdxs.end(), 0);
    sim.deallocateQubits(qubitIdxs);
  }
  // New one
  sim.allocateQubits(numQubits);
  // Prepare GHZ state: (distributed) |00...> + |11...>
  sim.h(0);
  for (int q = 0; q < numQubits - 1; ++q)
    sim.x({static_cast<std::size_t>(q)}, q + 1);
  sim.flushGateQueue();
  // Add the random state (kron)
  sim.allocateQubits(numAddedQubits, randState.get());
  sim.synchronize();
  auto finalState = sim.getSimulationState();
  std::vector<std::size_t> qubitIdxs(numQubits + numAddedQubits);
  std::iota(qubitIdxs.begin(), qubitIdxs.end(), 0);
  sim.deallocateQubits(qubitIdxs);
  std::vector<cudaq::complex> catState(1 << numQubits, 0.0);
  catState.front() = M_SQRT1_2;
  catState.back() = M_SQRT1_2;
  std::vector<cudaq::complex> addedState(1 << numAddedQubits);
  randState->toHost(addedState.data(), addedState.size());

  auto expectedState = kronProd(addedState, catState);
  std::vector<cudaq::complex> resultStateHost(expectedState.size());
  finalState->toHost(resultStateHost.data(), resultStateHost.size());
  for (std::size_t i = 0; i < resultStateHost.size(); ++i)
    EXPECT_NEAR(std::abs(resultStateHost[i] - expectedState[i]), 0.0,
                tolerance);
}

TEST(MGpuTesterMultiProcesses, CompatibleStateInputStaysOffHost) {
  auto &sim = getSimulator();
  int device = 0;
  HANDLE_CUDA_ERROR(cudaGetDevice(&device));
  auto communicator = std::make_shared<cudaq::cusv::CuStateVecCommunicator>(
      cudaq::cusv::CommunicatorPlugin::Auto, "libmpi.so");
  auto source = cudaq::cusv::CuStateVecState<cudaq::real>::createMultiProcess(
      numQubits, numQubits - 1, device,
      CUSTATEVEC_EX_MEMORY_SHARING_METHOD_NONE,
      {CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_COMMUNICATOR}, {1},
      std::size_t{1} << 26, std::move(communicator), true);
  source.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, numQubits - 1);
  source.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_GLOBAL_DEVICE, 1);
  source.setZeroState();
  source.synchronize();
  NoHostTransferState input(std::move(source));

  const auto qubits = sim.allocateQubits(numQubits, &input);
  sim.flushGateQueue();
  auto result = sim.getSimulationState();
  const std::vector<int> zeroState(numQubits, 0);
  EXPECT_NEAR(std::abs(result->getAmplitude(zeroState) -
                       std::complex<double>{1.0, 0.0}),
              0.0, tolerance);
  sim.deallocateQubits(qubits);
}

TEST(MGpuTesterMultiProcesses, ThresholdTransitionPreservesStateLayout) {
  auto &sim = getSimulator();
  const auto threshold = static_cast<std::size_t>(
      std::atoi(std::getenv("CUDAQ_MGPU_NQUBITS_THRESH")));
  ASSERT_GT(threshold, std::size_t{1});
  const auto belowThreshold = threshold - 1;
  std::vector<cudaq::complex> initial(1ULL << belowThreshold);
  initial.front() = M_SQRT1_2;
  initial.back() = M_SQRT1_2;
  sim.allocateQubits(belowThreshold, initial.data(), simulationPrecision);
  sim.allocateQubits(1);
  sim.synchronize();

  auto state = sim.getSimulationState();
  auto *const exState =
      dynamic_cast<cudaq::cusv::CuStateVecSimulationState<cudaq::real> *>(
          state.get());
  ASSERT_NE(exState, nullptr);
  const auto &descriptor = exState->state();
  EXPECT_EQ(descriptor.distributionType(),
            CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS);
  EXPECT_EQ(descriptor.numWires(), static_cast<int32_t>(threshold));
  EXPECT_EQ(descriptor.numLocalWires() + descriptor.numMigrationWires() + 1,
            descriptor.numWires());
  std::vector<cudaq::complex> actual(1ULL << threshold);
  state->toHost(actual.data(), actual.size());
  EXPECT_NEAR(std::abs(actual.front()), M_SQRT1_2, tolerance);
  EXPECT_NEAR(std::abs(actual[(1ULL << belowThreshold) - 1]), M_SQRT1_2,
              tolerance);
  std::vector<std::size_t> qubits(threshold);
  std::iota(qubits.begin(), qubits.end(), 0);
  sim.deallocateQubits(qubits);
}

TEST(MGpuTesterMultiProcesses, AddQubitsInStateSingleToMultiple) {
  auto &sim = getSimulator();
  const auto nQubitsMGPUThreshold = std::atoi(std::getenv(
      "CUDAQ_MGPU_NQUBITS_THRESH")); // This must be set to run this test
  EXPECT_TRUE(nQubitsMGPUThreshold > 0);
  const auto nQubits = nQubitsMGPUThreshold - 1;

  std::vector<cudaq::complex> state1 = randomState(nQubits); // Single-GPU state
  sim.allocateQubits(nQubits, state1.data(), simulationPrecision);
  // Another state: at this point, the state will become a multi-process state
  std::vector<cudaq::complex> state2 = randomState(nQubits);
  sim.allocateQubits(nQubits, state2.data(),
                     simulationPrecision); // Now become a multi-process state

  auto expectedState = kronProd(state2, state1);
  sim.synchronize();
  auto finalState = sim.getSimulationState();
  // Get a sub-state vector view (each rank sees different data)
  const auto numProcs = cudaq::mpi::num_ranks();
  const auto nTotalQubits = nQubits * 2;
  std::vector<cudaq::complex> subStateVec((1 << nTotalQubits) / numProcs);
  finalState->toHost(subStateVec.data(), subStateVec.size());
  // This check will be run on all ranks, i.e., all ranks now have a global view
  // of the state vector
  const auto rank = cudaq::mpi::rank();
  for (std::size_t i = 0; i < subStateVec.size(); ++i) {
    EXPECT_NEAR(
        std::abs(subStateVec[i] - expectedState[rank * subStateVec.size() + i]),
        0.0, tolerance);
  }
  std::vector<std::size_t> qubitIdxs(nTotalQubits);
  std::iota(qubitIdxs.begin(), qubitIdxs.end(), 0);
  sim.deallocateQubits(qubitIdxs);
}

// Test the case whereby we need to do reverse kron (the latter state is
// distributed then adding the original state)
TEST(MGpuTesterMultiProcesses, AddQubitsInStateSingleToMultipleUpper) {
  auto &sim = getSimulator();
  const auto nQubitsMGPUThreshold = std::atoi(std::getenv(
      "CUDAQ_MGPU_NQUBITS_THRESH")); // This must be set to run this test
  EXPECT_TRUE(nQubitsMGPUThreshold > 0);
  const auto nQubits = nQubitsMGPUThreshold;
  // Start out with just 1 qubit (not enough for any distribution)
  std::vector<cudaq::complex> state1 = randomState(1);
  sim.allocateQubits(1, state1.data(), simulationPrecision);
  // Another state: at this point, the state will become a multi-process state
  std::vector<cudaq::complex> state2 = randomState(nQubits);
  sim.allocateQubits(nQubits, state2.data(),
                     simulationPrecision); // Now become a multi-process state
  auto expectedState = kronProd(state2, state1);
  sim.synchronize();
  auto finalState = sim.getSimulationState();
  // Get a sub-state vector view (each rank sees different data)
  const auto numProcs = cudaq::mpi::num_ranks();
  const auto nTotalQubits = nQubits + 1;
  std::vector<cudaq::complex> subStateVec((1 << nTotalQubits) / numProcs);
  finalState->toHost(subStateVec.data(), subStateVec.size());
  // This check will be run on all ranks, i.e., all ranks now have a global view
  // of the state vector
  const auto rank = cudaq::mpi::rank();
  for (std::size_t i = 0; i < subStateVec.size(); ++i) {
    EXPECT_NEAR(
        std::abs(subStateVec[i] - expectedState[rank * subStateVec.size() + i]),
        0.0, tolerance);
  }
  std::vector<std::size_t> qubitIdxs(nTotalQubits);
  std::iota(qubitIdxs.begin(), qubitIdxs.end(), 0);
  sim.deallocateQubits(qubitIdxs);
}

TEST(MGpuTesterMultiProcesses, CommunicatorQueriesAreDescriptorLocal) {
  cudaq::cusv::CuStateVecCommunicator worldCommunicator(
      cudaq::cusv::CommunicatorPlugin::Auto, "libmpi.so");
  EXPECT_EQ(worldCommunicator.size(), 2);

  MPI_Comm singleton = MPI_COMM_NULL;
  ASSERT_EQ(
      MPI_Comm_split(MPI_COMM_WORLD, worldCommunicator.rank(), 0, &singleton),
      MPI_SUCCESS);
  cudaq::cusv::CuStateVecCommunicator singletonCommunicator(
      cudaq::cusv::CommunicatorPlugin::Auto, "libmpi.so");
  singletonCommunicator.setCommunicator(&singleton, sizeof(singleton));
  EXPECT_EQ(singletonCommunicator.size(), 1);
  EXPECT_EQ(singletonCommunicator.rank(), 0);
  EXPECT_EQ(worldCommunicator.size(), 2);

  auto *const plugin = cudaq::mpi::getMpiPlugin();
  const auto *const activeWorld = plugin->getComm();
  singletonCommunicator.setCommunicator(activeWorld->commPtr,
                                        activeWorld->commSize);
  EXPECT_EQ(MPI_Comm_free(&singleton), MPI_SUCCESS);
}

TEST(MGpuTesterMultiProcesses, FailedCommunicatorRebindingRestoresDescriptor) {
  cudaq::cusv::CuStateVecCommunicator communicator(
      cudaq::cusv::CommunicatorPlugin::Auto, "libmpi.so");
  const int32_t expectedSize = communicator.size();
  const int32_t expectedRank = communicator.rank();

  EXPECT_THROW(communicator.setCommunicator(nullptr, sizeof(MPI_Comm)),
               std::exception);
  EXPECT_EQ(communicator.size(), expectedSize);
  EXPECT_EQ(communicator.rank(), expectedRank);
}

TEST(MGpuTesterMultiProcesses, CommunicatorRebindingReinitializesState) {
  auto &sim = getSimulator();
  if (cudaq::mpi::num_ranks() != 4)
    GTEST_SKIP() << "This regression requires four MPI ranks.";
  MPI_Comm pair = MPI_COMM_NULL;
  ASSERT_EQ(MPI_Comm_split(MPI_COMM_WORLD, cudaq::mpi::rank() / 2, 0, &pair),
            MPI_SUCCESS);
  ASSERT_TRUE(sim.setMpiCommunicator(&pair, sizeof(pair)));
  auto *const plugin = cudaq::mpi::getMpiPlugin();
  ASSERT_TRUE(sim.setMpiCommunicator(plugin->getComm()->commPtr,
                                     plugin->getComm()->commSize));
  EXPECT_EQ(MPI_Comm_free(&pair), MPI_SUCCESS);
}

TEST(MGpuTesterMultiProcesses, CommunicatorRebindingPreservesUserSeed) {
  auto &sim = getSimulator();
  auto *const plugin = cudaq::mpi::getMpiPlugin();
  const auto *const activeWorld = plugin->getComm();
  MPI_Comm singleton = MPI_COMM_NULL;
  ASSERT_EQ(MPI_Comm_split(MPI_COMM_WORLD, cudaq::mpi::rank(), 0, &singleton),
            MPI_SUCCESS);

  constexpr std::size_t count = 100001;
  constexpr std::size_t seed = 1234;
  sim.setRandomSeed(seed);
  const auto expected = sim.generateLocalRandomNumbers(count);
  sim.setRandomSeed(seed);
  sim.allocateQubits(1);
  sim.deallocateQubits({0});
  ASSERT_TRUE(sim.setMpiCommunicator(&singleton, sizeof(singleton)));
  const auto actual = sim.generateLocalRandomNumbers(count);
  EXPECT_EQ(actual, expected);

  ASSERT_TRUE(
      sim.setMpiCommunicator(activeWorld->commPtr, activeWorld->commSize));
  EXPECT_EQ(MPI_Comm_free(&singleton), MPI_SUCCESS);
}

TEST(MGpuTesterMultiProcesses, CustomCommunicatorKeepsWorldDeviceRank) {
  auto &sim = getSimulator();
  auto *const plugin = cudaq::mpi::getMpiPlugin();
  MPI_Comm singleton = MPI_COMM_NULL;
  ASSERT_EQ(MPI_Comm_split(MPI_COMM_WORLD, cudaq::mpi::rank(), 0, &singleton),
            MPI_SUCCESS);
  ASSERT_TRUE(sim.setMpiCommunicator(&singleton, sizeof(singleton)));
  int deviceCount = 0;
  int device = 0;
  ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
  ASSERT_GT(deviceCount, 0);
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  EXPECT_EQ(device, cudaq::mpi::rank() % deviceCount);
  ASSERT_TRUE(sim.setMpiCommunicator(plugin->getComm()->commPtr,
                                     plugin->getComm()->commSize));
  EXPECT_EQ(MPI_Comm_free(&singleton), MPI_SUCCESS);
}

TEST(MGpuTesterMultiProcesses, CommunicatorRebindingLifecycle) {
  auto &sim = getSimulator();
  auto *const plugin = cudaq::mpi::getMpiPlugin();
  const auto *const activeWorld = plugin->getComm();

  MPI_Comm singleton = MPI_COMM_NULL;
  ASSERT_EQ(MPI_Comm_split(MPI_COMM_WORLD, cudaq::mpi::rank(), 0, &singleton),
            MPI_SUCCESS);
  sim.allocateQubits(1);
  EXPECT_THROW(sim.setMpiCommunicator(&singleton, sizeof(singleton)),
               std::runtime_error);
  sim.deallocateQubits({0});

  EXPECT_TRUE(sim.setMpiCommunicator(&singleton, sizeof(singleton)));
  EXPECT_TRUE(
      sim.setMpiCommunicator(activeWorld->commPtr, activeWorld->commSize));
  EXPECT_EQ(MPI_Comm_free(&singleton), MPI_SUCCESS);
}

TEST(MGpuTesterMultiProcesses,
     CommunicatorAllreduceSelectsInPlaceForAliasedBuffers) {
  auto *const plugin = cudaq::mpi::getMpiPlugin();
  auto *const distributed = plugin->get();
  const auto originalAllreduce = distributed->Allreduce;
  const auto originalAllreduceInPlace = distributed->AllreduceInPlace;
  distributed->Allreduce = spyAllreduce;
  distributed->AllreduceInPlace = spyAllreduceInPlace;

  allreduceCalls = 0;
  allreduceInPlaceCalls = 0;
  {
    cudaq::cusv::CuStateVecCommunicator communicator(
        cudaq::cusv::CommunicatorPlugin::Auto, "libmpi.so");
    double send = 1.0;
    double receive = 0.0;
    communicator.allReduce(&send, &receive, 1, CUDA_R_64F);
    EXPECT_EQ(allreduceCalls, 1);
    EXPECT_EQ(allreduceInPlaceCalls, 0);

    double inPlace = 1.0;
    communicator.allReduce(&inPlace, &inPlace, 1, CUDA_R_64F);
  }
  distributed->Allreduce = originalAllreduce;
  distributed->AllreduceInPlace = originalAllreduceInPlace;

  EXPECT_EQ(allreduceCalls, 1);
  EXPECT_EQ(allreduceInPlaceCalls, 1);
}
