/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecCircuitSimulatorEx.h"
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/PTSBESampler.h"
#include <cudaq.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <optional>

namespace {

class Environment {
public:
  Environment(const char *name, const char *value) : name_(name) {
    if (const char *old = std::getenv(name))
      old_ = old;
    setenv(name, value, 1);
  }

  ~Environment() {
    if (old_)
      setenv(name_, old_->c_str(), 1);
    else
      unsetenv(name_);
  }

private:
  const char *name_;
  std::optional<std::string> old_;
};

class ScopedExecutionContext {
public:
  ScopedExecutionContext(const char *name, std::size_t shots,
                         const cudaq::noise_model *noise = nullptr)
      : context_(name, shots) {
    context_.noiseModel = noise;
    cudaq::detail::setExecutionContext(&context_);
  }
  ~ScopedExecutionContext() { cudaq::detail::resetExecutionContext(); }

private:
  cudaq::ExecutionContext context_;
};

class SimulatorTester : public cudaq::cusv::CuStateVecCircuitSimulator<float> {
public:
  using Complex = std::complex<float>;
  using cudaq::cusv::CuStateVecCircuitSimulator<float>::generateRandomNumbers;

  void applyGateTask(const std::string &name,
                     const std::vector<Complex> &matrix,
                     const std::vector<float> &parameters = {}) {
    applyGate(GateApplicationTask(name, matrix, {}, {0}, parameters));
  }

  double nextCpuRandom() { return randomNumber(); }

  bool measure(std::size_t qubit) { return measureQubit(qubit); }

  custatevecExMatrixType_t deferredMatrixType() const {
    return std::get<cudaq::cusv::MatrixTask<float>>(m_deferredTasks.front())
        .matrixType;
  }

  using cudaq::cusv::CuStateVecCircuitSimulator<float>::makeNoiseTask;
  using cudaq::cusv::CuStateVecCircuitSimulator<float>::applyNoiseTask;
  using cudaq::cusv::CuStateVecCircuitSimulator<float>::observe;
  using cudaq::cusv::CuStateVecCircuitSimulator<float>::sample;
};

void checkMeasurementBeforeResetIsRejected() {
  constexpr std::size_t shots = 100;
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc();
  kernel.x(qubit);
  kernel.mz(qubit, "s0");
  kernel.reset(qubit);
  kernel.mz(qubit, "s1");

  // Measuring then resetting (and measuring again) is a mid-circuit
  // measurement. Non-explicit `sample` supports only terminal measurements, so
  // it must be rejected.
  cudaq::noise_model noise;
  EXPECT_ANY_THROW(cudaq::sample({.shots = shots, .noise = noise}, kernel));
}

cudaq::ptsbe::PTSBatch
makeDepolarizingBatch(const std::vector<std::size_t> &shots,
                      bool includeSequentialData = false) {
  cudaq::ptsbe::PTSBatch batch;
  batch.trace.emplace_back(cudaq::ptsbe::TraceInstructionType::Gate, "x",
                           std::vector<std::size_t>{0},
                           std::vector<std::size_t>{}, std::vector<double>{});
  batch.trace.emplace_back(
      cudaq::ptsbe::TraceInstructionType::Noise, "depolarization_channel",
      std::vector<std::size_t>{0}, std::vector<std::size_t>{},
      std::vector<double>{}, cudaq::depolarization_channel(0.75));
  batch.measureQubits = {0};
  batch.includeSequentialData = includeSequentialData;
  for (std::size_t branch = 0; branch < shots.size(); ++branch) {
    const cudaq::KrausSelection selection{
        1, {0}, "depolarization_channel", branch, branch != 0};
    batch.trajectories.emplace_back(
        branch, std::vector<cudaq::KrausSelection>{selection}, 0.25,
        shots[branch]);
  }
  return batch;
}

void checkDeterministicBranches(
    const std::vector<cudaq::sample_result> &results,
    const std::vector<std::size_t> &shots) {
  ASSERT_EQ(results.size(), shots.size());
  constexpr const char *expected[] = {"1", "0", "0", "1"};
  for (std::size_t branch = 0; branch < shots.size(); ++branch) {
    EXPECT_EQ(results[branch].get_total_shots(), shots[branch]);
    if (shots[branch] == 0)
      EXPECT_EQ(results[branch].size(), 0u);
    else
      EXPECT_EQ(results[branch].count(expected[branch]), shots[branch]);
  }
}

} // namespace

TEST(CuStateVecGateEngineTester, RejectsZeroObserveTrajectories) {
  SimulatorTester simulator;
  simulator.allocateQubits(1);
  cudaq::noise_model noise;
  cudaq::ExecutionContext context("observe");
  context.noiseModel = &noise;
  context.numberTrajectories = 0;
  cudaq::detail::setExecutionContext(&context);
  simulator.applyNoiseTask(cudaq::bit_flip_channel(0.5), {0});
  EXPECT_THROW(simulator.observe(cudaq::spin_op::z(0)), std::invalid_argument);
  cudaq::detail::resetExecutionContext();
  simulator.deallocateQubits({0});
}

TEST(CuStateVecGateEngineTester, DeferredNoiseUsesCompactMatrixTypes) {
  SimulatorTester simulator;
  const auto task = simulator.makeNoiseTask(cudaq::bit_flip_channel(0.5), {0});
  ASSERT_FALSE(task.matrixTypes.empty());
  for (const auto type : task.matrixTypes)
    EXPECT_NE(type, CUSTATEVEC_EX_MATRIX_DENSE);
}

TEST(CuStateVecGateEngineTester, MeasurementBeforeResetIsRejected) {
  checkMeasurementBeforeResetIsRejected();
}

TEST(CuStateVecPTSBETester, SimpleTrajectories) {
  Environment minimumBatch("CUDAQ_BATCHED_SIM_MIN_BATCH_SIZE", "1");
  SimulatorTester simulator;
  simulator.allocateQubits(1);

  cudaq::ptsbe::PTSBatch hadamard;
  hadamard.trace.emplace_back(cudaq::ptsbe::TraceInstructionType::Gate, "h",
                              std::vector<std::size_t>{0},
                              std::vector<std::size_t>{},
                              std::vector<double>{});
  hadamard.measureQubits = {0};
  hadamard.trajectories.emplace_back(0, std::vector<cudaq::KrausSelection>{},
                                     1.0, 1000);
  const auto hResult = simulator.sampleWithPTSBE(hadamard);
  ASSERT_EQ(hResult.size(), 1u);
  EXPECT_NEAR(hResult[0].probability("0"), 0.5, 0.1);
  EXPECT_NEAR(hResult[0].probability("1"), 0.5, 0.1);

  cudaq::ptsbe::PTSBatch bitFlip;
  bitFlip.trace.emplace_back(cudaq::ptsbe::TraceInstructionType::Gate, "x",
                             std::vector<std::size_t>{0},
                             std::vector<std::size_t>{}, std::vector<double>{});
  bitFlip.measureQubits = {0};
  bitFlip.trajectories.emplace_back(0, std::vector<cudaq::KrausSelection>{},
                                    1.0, 1000);
  const auto xResult = simulator.sampleWithPTSBE(bitFlip);
  ASSERT_EQ(xResult.size(), 1u);
  EXPECT_EQ(xResult[0].count("1"), 1000u);
  simulator.deallocateQubits({0});
}

TEST(CuStateVecPTSBETester, SameShotsPerTrajectory) {
  SimulatorTester simulator;
  simulator.allocateQubits(1);
  const std::vector<std::size_t> shots(4, 100);
  checkDeterministicBranches(
      simulator.sampleWithPTSBE(makeDepolarizingBatch(shots)), shots);
  simulator.deallocateQubits({0});
}

TEST(CuStateVecPTSBETester, DifferentAndZeroShotsPreserveOrder) {
  Environment minimumBatch("CUDAQ_BATCHED_SIM_MIN_BATCH_SIZE", "1");
  SimulatorTester simulator;
  simulator.allocateQubits(1);
  const std::vector<std::size_t> shots = {100, 0, 20, 50};
  checkDeterministicBranches(
      simulator.sampleWithPTSBE(makeDepolarizingBatch(shots)), shots);
  simulator.deallocateQubits({0});
}

TEST(CuStateVecPTSBETester, SequentialDataIsOptional) {
  SimulatorTester simulator;
  simulator.allocateQubits(1);
  const std::vector<std::size_t> shots(4, 10);
  const auto aggregated =
      simulator.sampleWithPTSBE(makeDepolarizingBatch(shots));
  const auto sequential =
      simulator.sampleWithPTSBE(makeDepolarizingBatch(shots, true));
  for (std::size_t branch = 0; branch < shots.size(); ++branch) {
    EXPECT_TRUE(aggregated[branch].sequential_data().empty());
    EXPECT_EQ(sequential[branch].sequential_data().size(), shots[branch]);
  }
  simulator.deallocateQubits({0});
}

TEST(CuStateVecPTSBETester, SequentialDataMultiQubit) {
  Environment minimumBatch("CUDAQ_BATCHED_SIM_MIN_BATCH_SIZE", "1");
  SimulatorTester simulator;
  simulator.allocateQubits(2);

  cudaq::ptsbe::PTSBatch batch;
  batch.trace.emplace_back(cudaq::ptsbe::TraceInstructionType::Gate, "h",
                           std::vector<std::size_t>{0},
                           std::vector<std::size_t>{}, std::vector<double>{});
  batch.trace.emplace_back(cudaq::ptsbe::TraceInstructionType::Gate, "x",
                           std::vector<std::size_t>{1},
                           std::vector<std::size_t>{0}, std::vector<double>{});
  batch.measureQubits = {0, 1};
  batch.includeSequentialData = true;
  batch.trajectories.emplace_back(0, std::vector<cudaq::KrausSelection>{}, 1.0,
                                  10);

  const auto results = simulator.sampleWithPTSBE(batch);
  ASSERT_EQ(results.size(), 1u);
  const auto sequential = results[0].sequential_data();
  ASSERT_EQ(sequential.size(), 10u);
  for (const auto &bits : sequential)
    EXPECT_TRUE(bits == "00" || bits == "11");
  simulator.deallocateQubits({0, 1});
}

TEST(CuStateVecPTSBETester, DeferredGatesRetainCompactMatrixType) {
  cudaq::noise_model noise;
  ScopedExecutionContext context("sample", 10, &noise);
  SimulatorTester simulator;
  simulator.allocateQubits(1);
  simulator.applyGateTask("z", {1.0f, 0.0f, 0.0f, -1.0f});
  EXPECT_EQ(simulator.deferredMatrixType(), CUSTATEVEC_EX_MATRIX_DIAGONAL);
  simulator.deallocateQubits({0});
}

TEST(CuStateVecPTSBETester, GpuRandomPathDoesNotAdvanceCpuEngine) {
  Environment threshold("CUDAQ_GPU_RNG_THRESHOLD", "0");
  SimulatorTester gpuSimulator;
  SimulatorTester referenceSimulator;
  gpuSimulator.setRandomSeed(42);
  referenceSimulator.setRandomSeed(42);

  const auto values = gpuSimulator.generateRandomNumbers(10000);
  ASSERT_EQ(values.size(), 10000u);
  EXPECT_EQ(gpuSimulator.nextCpuRandom(), referenceSimulator.nextCpuRandom());
}

TEST(CuStateVecPTSBETester, GpuRandomThresholdDispatch) {
  Environment threshold("CUDAQ_GPU_RNG_THRESHOLD", "0");
  SimulatorTester simulator;
  simulator.setRandomSeed(42);
  const auto values = simulator.generateRandomNumbers(10000);
  ASSERT_EQ(values.size(), 10000u);
  EXPECT_FALSE(std::is_sorted(values.begin(), values.end()));
  EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](double value) {
    return value >= 0.0 && value < 1.0;
  }));
}

TEST(CuStateVecPTSBETester, CpuRandomThresholdDispatch) {
  Environment threshold("CUDAQ_GPU_RNG_THRESHOLD", "999999999");
  SimulatorTester simulator;
  simulator.setRandomSeed(42);
  const auto values = simulator.generateRandomNumbers(10000);
  ASSERT_EQ(values.size(), 10000u);
  EXPECT_FALSE(std::is_sorted(values.begin(), values.end()));
  EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](double value) {
    return value >= 0.0 && value < 1.0;
  }));
}
struct xOp {
  void operator()() __qpu__ {
    cudaq::qvector q(1);
    x(q);
  }
};

struct bell {
  void operator()() __qpu__ {
    cudaq::qubit q, r;
    h(q);
    x<cudaq::ctrl>(q, r);
  }
};

TEST(PtsbeTest, checkDepolType) {
  for (double depolProb : {0.1, 0.25, 0.5, 0.75}) {
    for (std::size_t shots : {1024, 8192, 65536}) {
      printf("Testing depolarization with probability %f and shots %zu\n",
             depolProb, shots);
      cudaq::set_random_seed(13);
      cudaq::depolarization_channel depol(depolProb);
      cudaq::noise_model noise;
      noise.add_channel<cudaq::types::x>({0}, depol);
      auto counts_ptsbe = cudaq::ptsbe::sample(noise, shots, xOp{});
      counts_ptsbe.dump();
      EXPECT_EQ(2, counts_ptsbe.size());
      EXPECT_EQ(shots, counts_ptsbe.get_total_shots());
      auto counts = cudaq::sample({.shots = shots, .noise = noise}, xOp{});
      counts.dump();
      EXPECT_EQ(2, counts.size());

      // Check that the probabilities are close between ptsbe and regular
      // sampling with noise.
      const double tolerance =
          10.0 /
          std::sqrt(shots); // Statistical tolerance based on number of shots
      EXPECT_NEAR(counts_ptsbe.probability("0"), counts.probability("0"),
                  tolerance);
      EXPECT_NEAR(counts_ptsbe.probability("1"), counts.probability("1"),
                  tolerance);
    }
  }
}

TEST(PtsbeTest, checkDefinedNoiseModel) {
  cudaq::set_random_seed(13);
  cudaq::kraus_channel depol({cudaq::complex{0.99498743710662, 0.0},
                              {0.0, 0.0},
                              {0.0, 0.0},
                              {0.99498743710662, 0.0}},

                             {cudaq::complex{0.0, 0.0},
                              {0.05773502691896258, 0.0},
                              {0.05773502691896258, 0.0},
                              {0.0, 0.0}},

                             {cudaq::complex{0.0, 0.0},
                              {0.0, -0.05773502691896258},
                              {0.0, 0.05773502691896258},
                              {0.0, 0.0}},

                             {cudaq::complex{0.05773502691896258, 0.0},
                              {0.0, 0.0},
                              {0.0, 0.0},
                              {-0.05773502691896258, 0.0}});

  for (std::size_t shots : {1024, 8192, 65536}) {
    printf("Testing depolarization with %zu shots.\n", shots);
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_channel<cudaq::types::x>({0}, depol);
    auto counts_ptsbe = cudaq::ptsbe::sample(noise, shots, xOp{});
    counts_ptsbe.dump();
    EXPECT_EQ(2, counts_ptsbe.size());
    EXPECT_EQ(shots, counts_ptsbe.get_total_shots());
    auto counts = cudaq::sample({.shots = shots, .noise = noise}, xOp{});
    counts.dump();
    EXPECT_EQ(2, counts.size());

    // Check that the probabilities are close between ptsbe and regular
    // sampling with noise.
    const double tolerance =
        10.0 /
        std::sqrt(shots); // Statistical tolerance based on number of shots
    EXPECT_NEAR(counts_ptsbe.probability("0"), counts.probability("0"),
                tolerance);
    EXPECT_NEAR(counts_ptsbe.probability("1"), counts.probability("1"),
                tolerance);
  }
}

TEST(PtsbeTest, checkBitFlipType) {
  for (double prob : {0.1, 0.25, 0.5, 0.75}) {
    for (std::size_t shots : {8192, 65536}) {
      printf("Testing bit flip with probability %f and shots %zu\n", prob,
             shots);
      cudaq::set_random_seed(13);
      cudaq::bit_flip_channel bf(prob);
      cudaq::noise_model noise;
      noise.add_channel<cudaq::types::x>({0}, bf);
      auto counts_ptsbe = cudaq::ptsbe::sample(noise, shots, xOp{});
      counts_ptsbe.dump();
      const double tolerance =
          10.0 /
          std::sqrt(shots); // Statistical tolerance based on number of shots
      EXPECT_EQ(2, counts_ptsbe.size());
      EXPECT_EQ(shots, counts_ptsbe.get_total_shots());
      EXPECT_NEAR(counts_ptsbe.probability("0"), prob, tolerance);
      EXPECT_NEAR(counts_ptsbe.probability("1"), 1.0 - prob, tolerance);
    }
  }
}

TEST(PtsbeTest, checkCNOT) {
  for (double prob : {0.1, 0.25, 0.5, 0.75}) {
    for (std::size_t shots : {8192, 65536}) {
      printf(
          "Testing 2-qubit depolarization with probability %f and shots %zu\n",
          prob, shots);

      cudaq::set_random_seed(13);
      cudaq::depolarization2 depol(prob);
      cudaq::noise_model noise;
      noise.add_channel<cudaq::types::x>({0, 1}, depol);

      auto counts_ptsbe = cudaq::ptsbe::sample(noise, shots, bell{});
      counts_ptsbe.dump();
      EXPECT_EQ(4, counts_ptsbe.size());
      EXPECT_EQ(shots, counts_ptsbe.get_total_shots());
      auto counts = cudaq::sample({.shots = shots, .noise = noise}, bell{});
      counts.dump();
      EXPECT_EQ(4, counts.size());

      // Check that the probabilities are close between ptsbe and regular
      // sampling with noise.
      const double tolerance =
          10.0 /
          std::sqrt(shots); // Statistical tolerance based on number of shots
      for (const auto &bitStr : {"00", "01", "10", "11"}) {
        EXPECT_NEAR(counts_ptsbe.probability(bitStr),
                    counts.probability(bitStr), tolerance);
      }
    }
  }
}
