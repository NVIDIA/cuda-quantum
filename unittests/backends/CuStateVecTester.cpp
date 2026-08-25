/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "CuStateVecCircuitSimulatorEx.h"
#include "cudaq/ptsbe/PTSBESampler.h"
#include <cudaq.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <optional>
#include <type_traits>
#include <vector>

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

template <typename Scalar>
class CuStateVecCircuitSimulatorTester
    : public cudaq::cusv::CuStateVecCircuitSimulator<Scalar> {
  using Base = cudaq::cusv::CuStateVecCircuitSimulator<Scalar>;

public:
  using Complex = std::complex<Scalar>;
  using Base::generateRandomNumbers;

  void applyGateTask(const std::string &name,
                     const std::vector<Complex> &matrix,
                     const std::vector<Scalar> &parameters = {}) {
    this->applyGate(
        typename Base::GateApplicationTask(name, matrix, {}, {0}, parameters));
  }

  double nextCpuRandom() { return this->randomNumber(); }

  bool measure(std::size_t qubit) { return this->measureQubit(qubit); }

  custatevecExMatrixType_t deferredMatrixType() const {
    return std::get<cudaq::cusv::MatrixTask<Scalar>>(
               this->m_deferredTasks.front())
        .matrixType;
  }

  const std::vector<Complex> readStateVector() { return this->readState(); }

  bool stagedHostState = false;

  using Base::applyNoiseTask;
  using Base::makeNoiseTask;
  using Base::observe;
  using Base::sample;

protected:
  void writeState(const std::vector<Complex> &values) override {
    stagedHostState = true;
    Base::writeState(values);
  }
};

using SimulatorTester = CuStateVecCircuitSimulatorTester<cudaq::real>;
constexpr auto simulationPrecision = std::is_same_v<cudaq::real, float>
                                         ? cudaq::simulation_precision::fp32
                                         : cudaq::simulation_precision::fp64;

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

CUDAQ_TEST(CuStateVecCircuitSimulator, ImportsFullHostStateWithoutStaging) {
  using Complex = std::complex<cudaq::real>;
  const std::vector<Complex> expected{{0.0, 0.0}, {1.0, 0.0}};
  SimulatorTester simulator;

  simulator.allocateQubits(1, expected.data(), simulationPrecision);

  EXPECT_FALSE(simulator.stagedHostState);
  EXPECT_EQ(expected, simulator.readStateVector());
}

CUDAQ_TEST(CuStateVecGateEngineTester, RejectsZeroObserveTrajectories) {
  SimulatorTester simulator;
  simulator.allocateQubits(1);
  cudaq::noise_model noise;
  cudaq::ExecutionContext context("observe");
  context.noiseModel = &noise;
  context.numberTrajectories = 0;
  cudaq::detail::setExecutionContext(&context);
  simulator.applyNoiseTask(cudaq::bit_flip_channel(0.5), {0});
  EXPECT_ANY_THROW(simulator.observe(cudaq::spin_op::z(0)));
  cudaq::detail::resetExecutionContext();
  simulator.deallocateQubits({0});
}

CUDAQ_TEST(CuStateVecGateEngineTester, DeferredNoiseUsesCompactMatrixTypes) {
  SimulatorTester simulator;
  const auto task = simulator.makeNoiseTask(cudaq::bit_flip_channel(0.5), {0});
  ASSERT_FALSE(task.matrixTypes.empty());
  for (const auto type : task.matrixTypes)
    EXPECT_NE(type, CUSTATEVEC_EX_MATRIX_DENSE);
}

CUDAQ_TEST(CuStateVecGateEngineTester, MeasurementBeforeResetIsRejected) {
  checkMeasurementBeforeResetIsRejected();
}

CUDAQ_TEST(CuStateVecPTSBETester, SimpleTrajectories) {
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

CUDAQ_TEST(CuStateVecPTSBETester, SameShotsPerTrajectory) {
  SimulatorTester simulator;
  simulator.allocateQubits(1);
  const std::vector<std::size_t> shots(4, 100);
  checkDeterministicBranches(
      simulator.sampleWithPTSBE(makeDepolarizingBatch(shots)), shots);
  simulator.deallocateQubits({0});
}

CUDAQ_TEST(CuStateVecPTSBETester, DifferentAndZeroShotsPreserveOrder) {
  Environment minimumBatch("CUDAQ_BATCHED_SIM_MIN_BATCH_SIZE", "1");
  SimulatorTester simulator;
  simulator.allocateQubits(1);
  const std::vector<std::size_t> shots = {100, 0, 20, 50};
  checkDeterministicBranches(
      simulator.sampleWithPTSBE(makeDepolarizingBatch(shots)), shots);
  simulator.deallocateQubits({0});
}

CUDAQ_TEST(CuStateVecPTSBETester, SequentialDataIsOptional) {
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

CUDAQ_TEST(CuStateVecPTSBETester, SequentialDataMultiQubit) {
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

CUDAQ_TEST(CuStateVecPTSBETester, DeferredGatesRetainCompactMatrixType) {
  cudaq::noise_model noise;
  ScopedExecutionContext context("sample", 10, &noise);
  SimulatorTester simulator;
  simulator.allocateQubits(1);
  simulator.applyGateTask("z", {1.0f, 0.0f, 0.0f, -1.0f});
  EXPECT_EQ(simulator.deferredMatrixType(), CUSTATEVEC_EX_MATRIX_DIAGONAL);
  simulator.deallocateQubits({0});
}

CUDAQ_TEST(CuStateVecPTSBETester, GpuRandomPathDoesNotAdvanceCpuEngine) {
  Environment threshold("CUDAQ_GPU_RNG_THRESHOLD", "0");
  SimulatorTester gpuSimulator;
  SimulatorTester referenceSimulator;
  gpuSimulator.setRandomSeed(42);
  referenceSimulator.setRandomSeed(42);

  const auto values = gpuSimulator.generateRandomNumbers(10000);
  ASSERT_EQ(values.size(), 10000u);
  EXPECT_EQ(gpuSimulator.nextCpuRandom(), referenceSimulator.nextCpuRandom());
}

CUDAQ_TEST(CuStateVecPTSBETester, GpuRandomThresholdDispatch) {
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

CUDAQ_TEST(CuStateVecPTSBETester, CpuRandomThresholdDispatch) {
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
