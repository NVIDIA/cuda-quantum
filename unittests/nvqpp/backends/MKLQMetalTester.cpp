/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#define MKLQ_SIMULATOR_BACKEND_NAME "mklq_metal"
#define MKLQ_SIMULATOR_CLASS MklqMetalCircuitSimulator
#define MKLQ_SIMULATOR_PRINTED_NAME mklq_metal
#define MKLQ_SIMULATOR_DIAGNOSTIC_PREFIX "[mklq-metal]"
#define MKLQ_SIMULATOR_STATE_DIAGNOSTIC_PREFIX "[mklq-metal-state]"
#include "MklqCpuCircuitSimulator.cpp"
#include "MklqMetalRuntime.h"

#include "CUDAQTestUtils.h"

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

enum class ResidentFailureMode {
  None,
  SingleGate,
  TwoGate,
  Probability,
  Collapse,
  Reset
};

class MklqMetalCircuitSimulatorTester
    : public nvqir::MklqMetalCircuitSimulator {
public:
  void setResidentFailureModeForTest(ResidentFailureMode mode) {
    residentFailureMode = mode;
  }

  void setStateForTest(std::vector<std::complex<double>> data) {
#if defined(MKLQ_ENABLE_METAL_RUNTIME)
    invalidateMetalResidentState();
#endif
    nQubitsAllocated = std::countr_zero(data.size());
    previousStateDimension = stateDimension;
    stateDimension = data.size();
    state = std::move(data);
  }

  std::vector<double> fullRegisterProbabilitiesForTest() {
    std::vector<double> probabilities(state.size(), 0.0);
    fillFullRegisterProbabilities(probabilities);
    return probabilities;
  }

  cudaq::ExecutionResult sampleQubitsForTest(
      const std::vector<std::size_t> &qubits, int shots) {
    return sample(qubits, shots);
  }

  void applySingleQubitGateForTest(
      const std::vector<std::complex<double>> &matrix,
      const std::vector<std::size_t> &controls, std::size_t target) {
    applySingleQubitGate(matrix, controls, target, "", false);
  }

  void applyGateTaskForTest(const std::string &name,
                            const std::vector<std::complex<double>> &matrix,
                            const std::vector<std::size_t> &controls,
                            const std::vector<std::size_t> &targets) {
    nvqir::CircuitSimulatorBase<double>::GateApplicationTask task(
        name, matrix, controls, targets, {});
    applyGate(task);
  }

  bool measureQubitForTest(std::size_t qubit) { return measureQubit(qubit); }

  std::vector<std::complex<double>> stateVectorForTest() {
    auto simulationState = getSimulationState();
    std::vector<std::complex<double>> output(state.size());
    simulationState->toHost(output.data(), output.size());
    return output;
  }

  bool metalRuntimeAvailableForTest() const {
#if defined(MKLQ_ENABLE_METAL_RUNTIME)
    return metalExecutor.available();
#else
    return false;
#endif
  }

  std::size_t probabilityFillApplicationsForTest() const {
#if defined(MKLQ_ENABLE_METAL_RUNTIME)
    return metalExecutor.probabilityFillApplications();
#else
    return 0;
#endif
  }

  std::size_t measurementProbabilityApplicationsForTest() const {
#if defined(MKLQ_ENABLE_METAL_RUNTIME)
    return metalExecutor.measurementProbabilityApplications();
#else
    return 0;
#endif
  }

  std::size_t measurementProbabilityReductionApplicationsForTest() const {
#if defined(MKLQ_ENABLE_METAL_RUNTIME)
    return metalExecutor.measurementProbabilityReductionApplications();
#else
    return 0;
#endif
  }

  std::size_t measurementCollapseApplicationsForTest() const {
#if defined(MKLQ_ENABLE_METAL_RUNTIME)
    return metalExecutor.measurementCollapseApplications();
#else
    return 0;
#endif
  }

  std::size_t marginalProbabilityApplicationsForTest() const {
#if defined(MKLQ_ENABLE_METAL_RUNTIME)
    return metalExecutor.marginalProbabilityApplications();
#else
    return 0;
#endif
  }

  std::size_t singleQubitApplicationsForTest() const {
#if defined(MKLQ_ENABLE_METAL_RUNTIME)
    return metalExecutor.singleQubitGateApplications();
#else
    return 0;
#endif
  }

  std::size_t residentStateUploadsForTest() const {
#if defined(MKLQ_ENABLE_METAL_RUNTIME)
    return metalExecutor.residentStateUploads();
#else
    return 0;
#endif
  }

  std::size_t residentStateDownloadsForTest() const {
#if defined(MKLQ_ENABLE_METAL_RUNTIME)
    return metalExecutor.residentStateDownloads();
#else
    return 0;
#endif
  }

  std::size_t bitStringConversionsForTest() const {
    return bitStringConversions;
  }

protected:
#if defined(MKLQ_ENABLE_METAL_RUNTIME)
  bool applyMetalResidentSingleQubitGate(
      const std::complex<double> *matrix, const std::size_t *controlQubits,
      std::size_t controlCount, std::size_t target) override {
    if (residentFailureMode == ResidentFailureMode::SingleGate)
      return false;
    return nvqir::MklqMetalCircuitSimulator::applyMetalResidentSingleQubitGate(
        matrix, controlQubits, controlCount, target);
  }

  bool applyMetalResidentTwoQubitGate(
      const std::complex<double> *matrix, const std::size_t *controlQubits,
      std::size_t controlCount, const std::size_t *targets) override {
    if (residentFailureMode == ResidentFailureMode::TwoGate)
      return false;
    return nvqir::MklqMetalCircuitSimulator::applyMetalResidentTwoQubitGate(
        matrix, controlQubits, controlCount, targets);
  }

  bool computeMetalResidentMeasurementProbability(
      std::size_t qubit, double &probabilityOne) override {
    if (residentFailureMode == ResidentFailureMode::Probability)
      return false;
    return nvqir::MklqMetalCircuitSimulator::
        computeMetalResidentMeasurementProbability(qubit, probabilityOne);
  }

  bool collapseMetalResidentMeasurement(std::size_t qubit, bool result,
                                        double branchProbability) override {
    if (residentFailureMode == ResidentFailureMode::Collapse)
      return false;
    return nvqir::MklqMetalCircuitSimulator::collapseMetalResidentMeasurement(
        qubit, result, branchProbability);
  }

  bool applyMetalResidentResetGate(
      std::size_t qubit,
      const std::complex<double> *matrix) override {
    if (residentFailureMode == ResidentFailureMode::Reset)
      return false;
    return nvqir::MklqMetalCircuitSimulator::applyMetalResidentResetGate(
        qubit, matrix);
  }
#endif

private:
  ResidentFailureMode residentFailureMode = ResidentFailureMode::None;
};

void expectNear(std::complex<double> actual, std::complex<double> expected) {
  EXPECT_NEAR(actual.real(), expected.real(), 1.0e-6);
  EXPECT_NEAR(actual.imag(), expected.imag(), 1.0e-6);
}

bool expectMetalRuntimeReadyOrUnavailable(
    const nvqir::mklq::MetalStateVectorExecutor &executor) {
  if (executor.available())
    return true;

  const auto device = nvqir::mklq::queryMetalDevice();
  EXPECT_FALSE(device.available) << executor.lastError();
  return false;
}

} // namespace

CUDAQ_TEST(MKLQMetalTester, RegistersSeparateBackendName) {
  nvqir::MklqMetalCircuitSimulator sim;

  EXPECT_EQ(sim.name(), "mklq_metal");
  EXPECT_EQ(std::string(sim.clone()->name()), "mklq_metal");
}

CUDAQ_TEST(MKLQMetalTester, DiagnosticsUseMetalPrefix) {
  nvqir::MklqMetalCircuitSimulator sim;

  try {
    std::vector<std::complex<double>> amplitudes{{1.0, 0.0}, {0.0, 0.0},
                                                 {0.0, 0.0}};
    cudaq::state_data invalidState{amplitudes};
    (void)sim.createStateFromData(invalidState);
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("[mklq-metal]"),
              std::string::npos)
        << error.what();
    return;
  }

  FAIL() << "expected invalid metal state construction to raise";
}

CUDAQ_TEST(MKLQMetalTester, DetectsMetalRuntimeDevice) {
  const auto device = nvqir::mklq::queryMetalDevice();

  if (device.available)
    EXPECT_FALSE(device.name.empty());
  else
    EXPECT_TRUE(device.name.empty());
}

CUDAQ_TEST(MKLQMetalTester, MetalRuntimeAppliesSingleQubitGate) {
  nvqir::mklq::MetalStateVectorExecutor executor;

  if (!expectMetalRuntimeReadyOrUnavailable(executor))
    return;

  constexpr double invSqrt2 = 0.70710678118654752440;
  std::vector<std::complex<double>> state{{1.0, 0.0}, {0.0, 0.0}};
  const std::array<std::complex<double>, 4> hGate{
      std::complex<double>{invSqrt2, 0.0},
      std::complex<double>{invSqrt2, 0.0},
      std::complex<double>{invSqrt2, 0.0},
      std::complex<double>{-invSqrt2, 0.0}};

  ASSERT_TRUE(executor.applySingleQubitGate(state.data(), state.size(),
                                            hGate.data(), nullptr, 0, 0))
      << executor.lastError();

  expectNear(state[0], {invSqrt2, 0.0});
  expectNear(state[1], {invSqrt2, 0.0});
  EXPECT_EQ(executor.singleQubitGateApplications(), 1);
}

CUDAQ_TEST(MKLQMetalTester, MetalRuntimeAppliesControlledSingleQubitGate) {
  nvqir::mklq::MetalStateVectorExecutor executor;

  if (!expectMetalRuntimeReadyOrUnavailable(executor))
    return;

  std::vector<std::complex<double>> state{{0.0, 0.0},
                                          {0.0, 0.0},
                                          {1.0, 0.0},
                                          {0.0, 0.0}};
  const std::array<std::complex<double>, 4> xGate{
      std::complex<double>{0.0, 0.0}, std::complex<double>{1.0, 0.0},
      std::complex<double>{1.0, 0.0}, std::complex<double>{0.0, 0.0}};
  const std::array<std::size_t, 1> controls{1};

  ASSERT_TRUE(executor.applySingleQubitGate(state.data(), state.size(),
                                            xGate.data(), controls.data(),
                                            controls.size(), 0))
      << executor.lastError();

  expectNear(state[0], {0.0, 0.0});
  expectNear(state[1], {0.0, 0.0});
  expectNear(state[2], {0.0, 0.0});
  expectNear(state[3], {1.0, 0.0});
  EXPECT_EQ(executor.singleQubitGateApplications(), 1);
}

CUDAQ_TEST(MKLQMetalTester, MetalRuntimeAppliesTwoQubitGate) {
  nvqir::mklq::MetalStateVectorExecutor executor;

  if (!expectMetalRuntimeReadyOrUnavailable(executor))
    return;

  std::vector<std::complex<double>> state{{0.0, 0.0},
                                          {1.0, 0.0},
                                          {0.0, 0.0},
                                          {0.0, 0.0}};
  const std::array<std::complex<double>, 16> swapGate{
      std::complex<double>{1.0, 0.0}, std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0}, std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0}, std::complex<double>{0.0, 0.0},
      std::complex<double>{1.0, 0.0}, std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0}, std::complex<double>{1.0, 0.0},
      std::complex<double>{0.0, 0.0}, std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0}, std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0}, std::complex<double>{1.0, 0.0}};
  const std::array<std::size_t, 2> targets{0, 1};

  ASSERT_TRUE(executor.applyTwoQubitGate(state.data(), state.size(),
                                         swapGate.data(), nullptr, 0,
                                         targets.data()))
      << executor.lastError();

  expectNear(state[0], {0.0, 0.0});
  expectNear(state[1], {0.0, 0.0});
  expectNear(state[2], {1.0, 0.0});
  expectNear(state[3], {0.0, 0.0});
  EXPECT_EQ(executor.twoQubitGateApplications(), 1);
}

CUDAQ_TEST(MKLQMetalTester, MetalRuntimeAppliesControlledTwoQubitGate) {
  nvqir::mklq::MetalStateVectorExecutor executor;

  if (!expectMetalRuntimeReadyOrUnavailable(executor))
    return;

  std::vector<std::complex<double>> state{{0.0, 0.0}, {0.0, 0.0},
                                          {0.0, 0.0}, {0.0, 0.0},
                                          {0.0, 0.0}, {0.0, 0.0},
                                          {0.0, 0.0}, {1.0, 0.0}};
  const std::array<std::complex<double>, 16> czGate{
      std::complex<double>{1.0, 0.0},  std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0},  std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0},  std::complex<double>{1.0, 0.0},
      std::complex<double>{0.0, 0.0},  std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0},  std::complex<double>{0.0, 0.0},
      std::complex<double>{1.0, 0.0},  std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0},  std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0},  std::complex<double>{-1.0, 0.0}};
  const std::array<std::size_t, 1> controls{2};
  const std::array<std::size_t, 2> targets{0, 1};

  ASSERT_TRUE(executor.applyTwoQubitGate(state.data(), state.size(),
                                         czGate.data(), controls.data(),
                                         controls.size(), targets.data()))
      << executor.lastError();

  expectNear(state[7], {-1.0, 0.0});
  EXPECT_EQ(executor.twoQubitGateApplications(), 1);
}

CUDAQ_TEST(MKLQMetalTester, MetalRuntimeFillsFullRegisterProbabilities) {
  nvqir::mklq::MetalStateVectorExecutor executor;

  if (!expectMetalRuntimeReadyOrUnavailable(executor))
    return;

  std::vector<std::complex<double>> state{
      {3.0, 4.0}, {0.5, -0.5}, {0.0, 0.0}, {-2.0, 1.5}};
  std::vector<double> probabilities(state.size(), 0.0);

  ASSERT_TRUE(executor.fillFullRegisterProbabilities(
      state.data(), state.size(), probabilities.data(), probabilities.size()))
      << executor.lastError();

  EXPECT_DOUBLE_EQ(probabilities[0], 25.0);
  EXPECT_DOUBLE_EQ(probabilities[1], 0.5);
  EXPECT_DOUBLE_EQ(probabilities[2], 0.0);
  EXPECT_DOUBLE_EQ(probabilities[3], 6.25);
  EXPECT_EQ(executor.probabilityFillApplications(), 1);
}

CUDAQ_TEST(MKLQMetalTester, MetalRuntimeProbabilityFillMatchesCpuNorms) {
  nvqir::mklq::MetalStateVectorExecutor executor;

  if (!expectMetalRuntimeReadyOrUnavailable(executor))
    return;

  std::vector<std::complex<double>> state{
      {0.1, -0.2}, {-0.3, 0.125}, {1.0 / 3.0, -2.0 / 7.0},
      {-0.875, 0.0625}};
  std::vector<double> probabilities(state.size(), 0.0);

  ASSERT_TRUE(executor.fillFullRegisterProbabilities(
      state.data(), state.size(), probabilities.data(), probabilities.size()))
      << executor.lastError();

  for (std::size_t index = 0; index < state.size(); ++index)
    EXPECT_NEAR(probabilities[index], std::norm(state[index]), 1.0e-6)
        << "index " << index;
  EXPECT_EQ(executor.probabilityFillApplications(), 1);
}

CUDAQ_TEST(MKLQMetalTester,
           MetalRuntimeFillsResidentMarginalProbabilities) {
  nvqir::mklq::MetalStateVectorExecutor executor;

  if (!expectMetalRuntimeReadyOrUnavailable(executor))
    return;

  std::vector<std::complex<double>> state(8, {0.0, 0.0});
  state[0] = {std::sqrt(0.1), 0.0};
  state[1] = {std::sqrt(0.1), 0.0};
  state[3] = {std::sqrt(0.2), 0.0};
  state[4] = {std::sqrt(0.2), 0.0};
  state[5] = {std::sqrt(0.4), 0.0};

  const std::array<std::size_t, 2> qubits{2, 0};
  std::vector<double> probabilities(1ULL << qubits.size(), 0.0);

  ASSERT_TRUE(executor.uploadState(state.data(), state.size()))
      << executor.lastError();
  ASSERT_TRUE(executor.fillResidentMarginalProbabilities(
      qubits.data(), qubits.size(), probabilities.data(),
      probabilities.size()))
      << executor.lastError();

  ASSERT_EQ(probabilities.size(), 4);
  EXPECT_NEAR(probabilities[0], 0.1, 1.0e-6);
  EXPECT_NEAR(probabilities[1], 0.2, 1.0e-6);
  EXPECT_NEAR(probabilities[2], 0.3, 1.0e-6);
  EXPECT_NEAR(probabilities[3], 0.4, 1.0e-6);
  EXPECT_EQ(executor.marginalProbabilityApplications(), 1);
  EXPECT_EQ(executor.residentStateDownloads(), 0);
}

CUDAQ_TEST(MKLQMetalTester,
           MetalRuntimeComputesAndCollapsesResidentQubitProbability) {
  nvqir::mklq::MetalStateVectorExecutor executor;

  if (!expectMetalRuntimeReadyOrUnavailable(executor))
    return;

  std::vector<std::complex<double>> state{{0.5, 0.0},
                                          {0.0, 0.0},
                                          {std::sqrt(0.75), 0.0},
                                          {0.0, 0.0}};
  double probabilityOne = 0.0;

  ASSERT_TRUE(executor.uploadState(state.data(), state.size()))
      << executor.lastError();
  ASSERT_TRUE(executor.computeResidentQubitProbability(1, &probabilityOne))
      << executor.lastError();
  EXPECT_NEAR(probabilityOne, 0.75, 1.0e-6);

  ASSERT_TRUE(executor.collapseResidentQubit(1, true, probabilityOne))
      << executor.lastError();
  ASSERT_TRUE(executor.downloadState(state.data(), state.size()))
      << executor.lastError();

  expectNear(state[0], {0.0, 0.0});
  expectNear(state[1], {0.0, 0.0});
  expectNear(state[2], {1.0, 0.0});
  expectNear(state[3], {0.0, 0.0});
  EXPECT_EQ(executor.measurementProbabilityApplications(), 1);
  EXPECT_EQ(executor.measurementProbabilityReductionApplications(), 1);
  EXPECT_EQ(executor.probabilityFillApplications(), 0);
  EXPECT_EQ(executor.measurementCollapseApplications(), 1);
  EXPECT_EQ(executor.residentStateDownloads(), 1);
}

CUDAQ_TEST(MKLQMetalTester, MetalRuntimeKeepsResidentStateAcrossGateSequence) {
  nvqir::mklq::MetalStateVectorExecutor executor;

  if (!expectMetalRuntimeReadyOrUnavailable(executor))
    return;

  constexpr double invSqrt2 = 0.70710678118654752440;
  std::vector<std::complex<double>> state{{1.0, 0.0},
                                          {0.0, 0.0},
                                          {0.0, 0.0},
                                          {0.0, 0.0}};
  const std::array<std::complex<double>, 4> hGate{
      std::complex<double>{invSqrt2, 0.0},
      std::complex<double>{invSqrt2, 0.0},
      std::complex<double>{invSqrt2, 0.0},
      std::complex<double>{-invSqrt2, 0.0}};
  const std::array<std::complex<double>, 4> xGate{
      std::complex<double>{0.0, 0.0}, std::complex<double>{1.0, 0.0},
      std::complex<double>{1.0, 0.0}, std::complex<double>{0.0, 0.0}};
  const std::array<std::size_t, 1> controls{0};

  ASSERT_TRUE(executor.uploadState(state.data(), state.size()))
      << executor.lastError();
  ASSERT_TRUE(executor.applyResidentSingleQubitGate(hGate.data(), nullptr, 0,
                                                    0))
      << executor.lastError();
  ASSERT_TRUE(executor.applyResidentSingleQubitGate(xGate.data(),
                                                    controls.data(),
                                                    controls.size(), 1))
      << executor.lastError();
  ASSERT_TRUE(executor.downloadState(state.data(), state.size()))
      << executor.lastError();

  expectNear(state[0], {invSqrt2, 0.0});
  expectNear(state[1], {0.0, 0.0});
  expectNear(state[2], {0.0, 0.0});
  expectNear(state[3], {invSqrt2, 0.0});
  EXPECT_EQ(executor.residentStateUploads(), 1);
  EXPECT_EQ(executor.residentStateDownloads(), 1);
  EXPECT_EQ(executor.singleQubitGateApplications(), 2);
}

CUDAQ_TEST(MKLQMetalTester,
           MetalRuntimeKeepsResidentYAndControlledYSequence) {
  nvqir::mklq::MetalStateVectorExecutor executor;

  if (!expectMetalRuntimeReadyOrUnavailable(executor))
    return;

  std::vector<std::complex<double>> state{{1.0, 0.0},
                                          {0.0, 0.0},
                                          {0.0, 0.0},
                                          {0.0, 0.0}};
  const std::array<std::complex<double>, 4> yGate{
      std::complex<double>{0.0, 0.0}, std::complex<double>{0.0, -1.0},
      std::complex<double>{0.0, 1.0}, std::complex<double>{0.0, 0.0}};
  const std::array<std::size_t, 1> controls{0};

  ASSERT_TRUE(executor.uploadState(state.data(), state.size()))
      << executor.lastError();
  ASSERT_TRUE(executor.applyResidentSingleQubitGate(yGate.data(), nullptr, 0,
                                                    0))
      << executor.lastError();
  ASSERT_TRUE(executor.applyResidentSingleQubitGate(yGate.data(),
                                                    controls.data(),
                                                    controls.size(), 1))
      << executor.lastError();
  ASSERT_TRUE(executor.downloadState(state.data(), state.size()))
      << executor.lastError();

  expectNear(state[0], {0.0, 0.0});
  expectNear(state[1], {0.0, 0.0});
  expectNear(state[2], {0.0, 0.0});
  expectNear(state[3], {-1.0, 0.0});
  EXPECT_EQ(executor.residentStateUploads(), 1);
  EXPECT_EQ(executor.residentStateDownloads(), 1);
  EXPECT_EQ(executor.singleQubitGateApplications(), 2);
}

CUDAQ_TEST(MKLQMetalTester,
           MetalRuntimeFillsResidentProbabilitiesWithoutStateReadback) {
  nvqir::mklq::MetalStateVectorExecutor executor;

  if (!expectMetalRuntimeReadyOrUnavailable(executor))
    return;

  constexpr double invSqrt2 = 0.70710678118654752440;
  std::vector<std::complex<double>> state{{invSqrt2, 0.0},
                                          {0.0, 0.0},
                                          {0.0, 0.0},
                                          {invSqrt2, 0.0}};
  std::vector<double> probabilities(state.size(), 0.0);

  ASSERT_TRUE(executor.uploadState(state.data(), state.size()))
      << executor.lastError();
  ASSERT_TRUE(executor.fillResidentFullRegisterProbabilities(
      probabilities.data(), probabilities.size()))
      << executor.lastError();

  EXPECT_NEAR(probabilities[0], 0.5, 1.0e-6);
  EXPECT_NEAR(probabilities[1], 0.0, 1.0e-6);
  EXPECT_NEAR(probabilities[2], 0.0, 1.0e-6);
  EXPECT_NEAR(probabilities[3], 0.5, 1.0e-6);
  EXPECT_EQ(executor.residentStateUploads(), 1);
  EXPECT_EQ(executor.residentStateDownloads(), 0);
  EXPECT_EQ(executor.probabilityFillApplications(), 1);
}

CUDAQ_TEST(MKLQMetalTester, MetalRuntimeRejectsTargetsOutsideStateRange) {
  nvqir::mklq::MetalStateVectorExecutor executor;

  if (!expectMetalRuntimeReadyOrUnavailable(executor))
    return;

  std::vector<std::complex<double>> state{{1.0, 0.0}, {0.0, 0.0}};
  const std::array<std::complex<double>, 4> xGate{
      std::complex<double>{0.0, 0.0}, std::complex<double>{1.0, 0.0},
      std::complex<double>{1.0, 0.0}, std::complex<double>{0.0, 0.0}};
  const std::array<std::size_t, 1> invalidControls{2};
  const std::array<std::size_t, 1> overlappingControl{0};
  const std::array<std::size_t, 2> duplicateControls{1, 1};
  const std::array<std::complex<double>, 16> identityTwoQubitGate{
      std::complex<double>{1.0, 0.0}, std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0}, std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0}, std::complex<double>{1.0, 0.0},
      std::complex<double>{0.0, 0.0}, std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0}, std::complex<double>{0.0, 0.0},
      std::complex<double>{1.0, 0.0}, std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0}, std::complex<double>{0.0, 0.0},
      std::complex<double>{0.0, 0.0}, std::complex<double>{1.0, 0.0}};
  const std::array<std::size_t, 2> invalidTwoQubitTargets{0, 2};
  const std::array<std::size_t, 2> validTwoQubitTargets{0, 1};

  EXPECT_FALSE(executor.applySingleQubitGate(state.data(), state.size(),
                                             xGate.data(), nullptr, 0, 2));
  EXPECT_FALSE(executor.applySingleQubitGate(
      state.data(), state.size(), xGate.data(), invalidControls.data(),
      invalidControls.size(), 0));
  EXPECT_FALSE(executor.applySingleQubitGate(
      state.data(), state.size(), xGate.data(), overlappingControl.data(),
      overlappingControl.size(), 0));
  EXPECT_FALSE(executor.applySingleQubitGate(
      state.data(), state.size(), xGate.data(), duplicateControls.data(),
      duplicateControls.size(), 0));
  std::vector<std::complex<double>> twoQubitState{
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  EXPECT_FALSE(executor.applyTwoQubitGate(
      twoQubitState.data(), twoQubitState.size(), identityTwoQubitGate.data(),
      nullptr, 0, invalidTwoQubitTargets.data()));
  EXPECT_FALSE(executor.applyTwoQubitGate(
      twoQubitState.data(), twoQubitState.size(), identityTwoQubitGate.data(),
      invalidControls.data(), invalidControls.size(),
      validTwoQubitTargets.data()));
  EXPECT_FALSE(executor.applyTwoQubitGate(
      twoQubitState.data(), twoQubitState.size(), identityTwoQubitGate.data(),
      overlappingControl.data(), overlappingControl.size(),
      validTwoQubitTargets.data()));
  EXPECT_FALSE(executor.applyTwoQubitGate(
      twoQubitState.data(), twoQubitState.size(), identityTwoQubitGate.data(),
      duplicateControls.data(), duplicateControls.size(),
      validTwoQubitTargets.data()));

  ASSERT_TRUE(executor.uploadState(state.data(), state.size()))
      << executor.lastError();
  EXPECT_FALSE(
      executor.applyResidentSingleQubitGate(xGate.data(), nullptr, 0, 2));
  EXPECT_FALSE(executor.applyResidentSingleQubitGate(
      xGate.data(), invalidControls.data(), invalidControls.size(), 0));
  EXPECT_FALSE(executor.applyResidentSingleQubitGate(
      xGate.data(), overlappingControl.data(), overlappingControl.size(), 0));
  EXPECT_FALSE(executor.applyResidentSingleQubitGate(
      xGate.data(), duplicateControls.data(), duplicateControls.size(), 0));

  ASSERT_TRUE(executor.uploadState(twoQubitState.data(), twoQubitState.size()))
      << executor.lastError();
  EXPECT_FALSE(executor.applyResidentTwoQubitGate(
      identityTwoQubitGate.data(), nullptr, 0, invalidTwoQubitTargets.data()));
  EXPECT_FALSE(executor.applyResidentTwoQubitGate(
      identityTwoQubitGate.data(), invalidControls.data(),
      invalidControls.size(), validTwoQubitTargets.data()));
  EXPECT_FALSE(executor.applyResidentTwoQubitGate(
      identityTwoQubitGate.data(), overlappingControl.data(),
      overlappingControl.size(), validTwoQubitTargets.data()));
  EXPECT_FALSE(executor.applyResidentTwoQubitGate(
      identityTwoQubitGate.data(), duplicateControls.data(),
      duplicateControls.size(), validTwoQubitTargets.data()));
}

CUDAQ_TEST(MKLQMetalTester, SimulatorUsesMetalFullRegisterProbabilityFill) {
  MklqMetalCircuitSimulatorTester sim;
  sim.setStateForTest({{3.0, 4.0}, {0.5, -0.5}, {0.0, 0.0}, {-2.0, 1.5}});

  const auto probabilities = sim.fullRegisterProbabilitiesForTest();

  ASSERT_EQ(probabilities.size(), 4);
  EXPECT_DOUBLE_EQ(probabilities[0], 25.0);
  EXPECT_DOUBLE_EQ(probabilities[1], 0.5);
  EXPECT_DOUBLE_EQ(probabilities[2], 0.0);
  EXPECT_DOUBLE_EQ(probabilities[3], 6.25);
  EXPECT_EQ(sim.probabilityFillApplicationsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorKeepsSupportedGateSequenceResidentUntilReadback) {
  constexpr double invSqrt2 = 0.70710678118654752440;
  const std::vector<std::complex<double>> hGate{
      {invSqrt2, 0.0}, {invSqrt2, 0.0}, {invSqrt2, 0.0}, {-invSqrt2, 0.0}};
  const std::vector<std::complex<double>> xGate{{0.0, 0.0},
                                                {1.0, 0.0},
                                                {1.0, 0.0},
                                                {0.0, 0.0}};

  MklqMetalCircuitSimulatorTester sim;
  sim.setStateForTest(
      {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}});
  sim.applySingleQubitGateForTest(hGate, {}, 0);
  sim.applySingleQubitGateForTest(xGate, {0}, 1);

  const auto state = sim.stateVectorForTest();

  ASSERT_EQ(state.size(), 4);
  expectNear(state[0], {invSqrt2, 0.0});
  expectNear(state[1], {0.0, 0.0});
  expectNear(state[2], {0.0, 0.0});
  expectNear(state[3], {invSqrt2, 0.0});
  EXPECT_EQ(sim.singleQubitApplicationsForTest(),
            sim.metalRuntimeAvailableForTest() ? 2 : 0);
  EXPECT_EQ(sim.residentStateUploadsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
  EXPECT_EQ(sim.residentStateDownloadsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorKeepsYAndControlledYResidentUntilReadback) {
  const std::vector<std::complex<double>> yGate{{0.0, 0.0},
                                                {0.0, -1.0},
                                                {0.0, 1.0},
                                                {0.0, 0.0}};

  MklqMetalCircuitSimulatorTester sim;
  sim.setStateForTest(
      {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}});
  sim.applySingleQubitGateForTest(yGate, {}, 0);
  sim.applySingleQubitGateForTest(yGate, {0}, 1);

  EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
  const auto state = sim.stateVectorForTest();

  ASSERT_EQ(state.size(), 4);
  expectNear(state[0], {0.0, 0.0});
  expectNear(state[1], {0.0, 0.0});
  expectNear(state[2], {0.0, 0.0});
  expectNear(state[3], {-1.0, 0.0});
  EXPECT_EQ(sim.singleQubitApplicationsForTest(),
            sim.metalRuntimeAvailableForTest() ? 2 : 0);
  EXPECT_EQ(sim.residentStateUploadsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
  EXPECT_EQ(sim.residentStateDownloadsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorKeepsBuiltInYAndControlledYResidentUntilReadback) {
  MklqMetalCircuitSimulatorTester sim;
  sim.setStateForTest(
      {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}});
  sim.y(0);
  sim.y({0}, 1);
  sim.flushGateQueue();

  EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
  const auto state = sim.stateVectorForTest();

  ASSERT_EQ(state.size(), 4);
  expectNear(state[0], {0.0, 0.0});
  expectNear(state[1], {0.0, 0.0});
  expectNear(state[2], {0.0, 0.0});
  expectNear(state[3], {-1.0, 0.0});
  EXPECT_EQ(sim.singleQubitApplicationsForTest(),
            sim.metalRuntimeAvailableForTest() ? 2 : 0);
  EXPECT_EQ(sim.residentStateUploadsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
  EXPECT_EQ(sim.residentStateDownloadsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorKeepsBuiltInRzAndControlledRzResidentUntilReadback) {
  constexpr double theta = 0.375;
  constexpr double phi = -0.8125;
  const std::vector<std::complex<double>> initial{
      {0.25, 0.5}, {-0.125, 0.375}, {0.5, -0.25}, {-0.75, 0.125}};

  auto rzPhase = [](double angle, bool oneBranch) {
    const auto phase = oneBranch ? angle / 2.0 : -angle / 2.0;
    return std::complex<double>{std::cos(phase), std::sin(phase)};
  };

  auto expected = initial;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const bool q0IsOne = (index & 1ULL) != 0;
    const bool q1IsOne = (index & 2ULL) != 0;
    expected[index] *= rzPhase(theta, q0IsOne);
    if (q0IsOne)
      expected[index] *= rzPhase(phi, q1IsOne);
  }

  MklqMetalCircuitSimulatorTester sim;
  sim.setStateForTest(initial);
  sim.rz(theta, 0);
  sim.rz(phi, std::vector<std::size_t>{0}, 1);
  sim.flushGateQueue();

  EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
  const auto state = sim.stateVectorForTest();

  ASSERT_EQ(state.size(), expected.size());
  for (std::size_t index = 0; index < state.size(); ++index)
    expectNear(state[index], expected[index]);
  EXPECT_EQ(sim.singleQubitApplicationsForTest(),
            sim.metalRuntimeAvailableForTest() ? 2 : 0);
  EXPECT_EQ(sim.residentStateUploadsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
  EXPECT_EQ(sim.residentStateDownloadsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorSamplesResidentDenseStateWithoutReadback) {
  constexpr std::size_t qubitCount = 7;
  constexpr std::size_t dimension = 1ULL << qubitCount;
  constexpr double invSqrt2 = 0.70710678118654752440;
  const std::vector<std::complex<double>> hGate{
      {invSqrt2, 0.0}, {invSqrt2, 0.0}, {invSqrt2, 0.0}, {-invSqrt2, 0.0}};

  MklqMetalCircuitSimulatorTester sim;
  std::vector<std::complex<double>> state(dimension, {0.0, 0.0});
  state[0] = {1.0, 0.0};
  sim.setStateForTest(std::move(state));

  std::vector<std::size_t> qubits;
  qubits.reserve(qubitCount);
  for (std::size_t qubit = 0; qubit < qubitCount; ++qubit) {
    qubits.push_back(qubit);
    sim.applySingleQubitGateForTest(hGate, {}, qubit);
  }

  const auto counts = sim.sampleQubitsForTest(qubits, 8);
  std::size_t totalShots = 0;
  for (const auto &[bits, count] : counts.counts)
    totalShots += count;

  EXPECT_EQ(totalShots, 8);
  EXPECT_EQ(sim.singleQubitApplicationsForTest(),
            sim.metalRuntimeAvailableForTest() ? qubitCount : 0);
  EXPECT_EQ(sim.residentStateUploadsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
  EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
  EXPECT_EQ(sim.probabilityFillApplicationsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorSamplesLargeResidentPartialRegisterThroughFullProbability) {
  constexpr std::size_t qubitCount = 16;
  constexpr std::size_t dimension = 1ULL << qubitCount;
  const std::vector<std::complex<double>> xGate{{0.0, 0.0},
                                                {1.0, 0.0},
                                                {1.0, 0.0},
                                                {0.0, 0.0}};

  MklqMetalCircuitSimulatorTester sim;
  std::vector<std::complex<double>> state(dimension, {0.0, 0.0});
  state[0] = {1.0, 0.0};
  sim.setStateForTest(std::move(state));
  sim.applySingleQubitGateForTest(xGate, {}, 0);
  sim.applySingleQubitGateForTest(xGate, {}, 14);

  const std::vector<std::size_t> measuredQubits{0, 2, 4, 6, 8, 10, 12, 14};
  const auto counts = sim.sampleQubitsForTest(measuredQubits, 8);

  ASSERT_EQ(counts.counts.size(), 1);
  ASSERT_TRUE(counts.counts.contains("10000001"));
  EXPECT_EQ(counts.counts.at("10000001"), 8);
  EXPECT_EQ(sim.singleQubitApplicationsForTest(),
            sim.metalRuntimeAvailableForTest() ? 2 : 0);
  EXPECT_EQ(sim.residentStateUploadsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
  EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
  EXPECT_EQ(sim.marginalProbabilityApplicationsForTest(), 0);
  EXPECT_EQ(sim.probabilityFillApplicationsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorSamplesSmallResidentPartialRegisterThroughMarginalProbability) {
  constexpr std::size_t qubitCount = 7;
  constexpr std::size_t dimension = 1ULL << qubitCount;
  const std::vector<std::complex<double>> xGate{{0.0, 0.0},
                                                {1.0, 0.0},
                                                {1.0, 0.0},
                                                {0.0, 0.0}};

  MklqMetalCircuitSimulatorTester sim;
  std::vector<std::complex<double>> state(dimension, {0.0, 0.0});
  state[0] = {1.0, 0.0};
  sim.setStateForTest(std::move(state));
  sim.applySingleQubitGateForTest(xGate, {}, 0);
  sim.applySingleQubitGateForTest(xGate, {}, 4);

  const std::vector<std::size_t> measuredQubits{0, 2, 4};
  const auto counts = sim.sampleQubitsForTest(measuredQubits, 8);

  ASSERT_EQ(counts.counts.size(), 1);
  ASSERT_TRUE(counts.counts.contains("101"));
  EXPECT_EQ(counts.counts.at("101"), 8);
  EXPECT_EQ(sim.singleQubitApplicationsForTest(),
            sim.metalRuntimeAvailableForTest() ? 2 : 0);
  EXPECT_EQ(sim.residentStateUploadsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
  EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
  EXPECT_EQ(sim.marginalProbabilityApplicationsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
  EXPECT_EQ(sim.probabilityFillApplicationsForTest(), 0);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorSamplesDeterministicSparseStateWithOneBitStringConversion) {
  std::vector<std::complex<double>> state(8, {0.0, 0.0});
  state[5] = {1.0, 0.0};

  MklqMetalCircuitSimulatorTester sim;
  sim.setStateForTest(std::move(state));

  constexpr int shots = 32;
  const auto counts = sim.sampleQubitsForTest({0, 1, 2}, shots);

  ASSERT_EQ(counts.counts.size(), 1);
  ASSERT_TRUE(counts.counts.contains("101"));
  EXPECT_EQ(counts.counts.at("101"), shots);
  EXPECT_EQ(counts.sequentialData.size(), shots);
  EXPECT_EQ(sim.bitStringConversionsForTest(), 1);
  EXPECT_EQ(sim.probabilityFillApplicationsForTest(), 0);
  EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorSynchronizesResidentStateBeforeUnsupportedGate) {
  constexpr double invSqrt2 = 0.70710678118654752440;
  const std::vector<std::complex<double>> hGate{
      {invSqrt2, 0.0}, {invSqrt2, 0.0}, {invSqrt2, 0.0}, {-invSqrt2, 0.0}};
  std::vector<std::complex<double>> identityThreeQubit(64, {0.0, 0.0});
  for (std::size_t index = 0; index < 8; ++index)
    identityThreeQubit[index * 8 + index] = {1.0, 0.0};

  MklqMetalCircuitSimulatorTester sim;
  std::vector<std::complex<double>> state(8, {0.0, 0.0});
  state[0] = {1.0, 0.0};
  sim.setStateForTest(std::move(state));

  sim.applySingleQubitGateForTest(hGate, {}, 0);
  sim.applyGateTaskForTest("identity3", identityThreeQubit, {}, {0, 1, 2});
  const auto output = sim.stateVectorForTest();

  ASSERT_EQ(output.size(), 8);
  expectNear(output[0], {invSqrt2, 0.0});
  expectNear(output[1], {invSqrt2, 0.0});
  for (std::size_t index = 2; index < output.size(); ++index)
    expectNear(output[index], {0.0, 0.0});
  EXPECT_EQ(sim.residentStateDownloadsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorMeasuresAndResetsResidentStateWithoutReadback) {
  const std::vector<std::complex<double>> xGate{{0.0, 0.0},
                                                {1.0, 0.0},
                                                {1.0, 0.0},
                                                {0.0, 0.0}};

  MklqMetalCircuitSimulatorTester measured;
  measured.setStateForTest({{1.0, 0.0}, {0.0, 0.0}});
  measured.applySingleQubitGateForTest(xGate, {}, 0);
  EXPECT_TRUE(measured.measureQubitForTest(0));
  EXPECT_EQ(measured.residentStateDownloadsForTest(),
            0);
  EXPECT_EQ(measured.measurementProbabilityApplicationsForTest(),
            measured.metalRuntimeAvailableForTest() ? 1 : 0);
  EXPECT_EQ(measured.measurementProbabilityReductionApplicationsForTest(),
            measured.metalRuntimeAvailableForTest() ? 1 : 0);
  EXPECT_EQ(measured.measurementCollapseApplicationsForTest(),
            measured.metalRuntimeAvailableForTest() ? 1 : 0);

  MklqMetalCircuitSimulatorTester reset;
  reset.setStateForTest({{1.0, 0.0}, {0.0, 0.0}});
  reset.applySingleQubitGateForTest(xGate, {}, 0);
  reset.resetQubit(0);
  EXPECT_EQ(reset.residentStateDownloadsForTest(), 0);
  EXPECT_EQ(reset.measurementProbabilityApplicationsForTest(),
            reset.metalRuntimeAvailableForTest() ? 1 : 0);
  EXPECT_EQ(reset.measurementProbabilityReductionApplicationsForTest(),
            reset.metalRuntimeAvailableForTest() ? 1 : 0);
  EXPECT_EQ(reset.measurementCollapseApplicationsForTest(),
            reset.metalRuntimeAvailableForTest() ? 1 : 0);
  const auto output = reset.stateVectorForTest();

  ASSERT_EQ(output.size(), 2);
  expectNear(output[0], {1.0, 0.0});
  expectNear(output[1], {0.0, 0.0});
  EXPECT_EQ(reset.residentStateDownloadsForTest(),
            reset.metalRuntimeAvailableForTest() ? 1 : 0);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorResetsResidentNonzeroTargetWithoutReadback) {
  constexpr double invSqrt2 = 0.70710678118654752440;
  const std::vector<std::complex<double>> hGate{
      {invSqrt2, 0.0}, {invSqrt2, 0.0}, {invSqrt2, 0.0}, {-invSqrt2, 0.0}};

  MklqMetalCircuitSimulatorTester reset;
  reset.setStateForTest(
      {{0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}});
  reset.applySingleQubitGateForTest(hGate, {}, 0);
  reset.resetQubit(1);

  EXPECT_EQ(reset.residentStateDownloadsForTest(), 0);
  EXPECT_EQ(reset.measurementProbabilityApplicationsForTest(),
            reset.metalRuntimeAvailableForTest() ? 1 : 0);
  EXPECT_EQ(reset.measurementCollapseApplicationsForTest(),
            reset.metalRuntimeAvailableForTest() ? 1 : 0);

  const auto output = reset.stateVectorForTest();
  ASSERT_EQ(output.size(), 4);
  expectNear(output[0], {invSqrt2, 0.0});
  expectNear(output[1], {invSqrt2, 0.0});
  expectNear(output[2], {0.0, 0.0});
  expectNear(output[3], {0.0, 0.0});
  EXPECT_EQ(reset.residentStateDownloadsForTest(),
            reset.metalRuntimeAvailableForTest() ? 1 : 0);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorPoisonsResidentStateWhenSingleGateFails) {
  const std::vector<std::complex<double>> xGate{{0.0, 0.0},
                                                {1.0, 0.0},
                                                {1.0, 0.0},
                                                {0.0, 0.0}};

  MklqMetalCircuitSimulatorTester sim;
  if (!sim.metalRuntimeAvailableForTest())
    return;

  sim.setStateForTest({{1.0, 0.0}, {0.0, 0.0}});
  sim.applySingleQubitGateForTest(xGate, {}, 0);
  ASSERT_EQ(sim.residentStateUploadsForTest(), 1);
  ASSERT_EQ(sim.singleQubitApplicationsForTest(), 1);
  ASSERT_EQ(sim.residentStateDownloadsForTest(), 0);
  sim.setResidentFailureModeForTest(ResidentFailureMode::SingleGate);

  try {
    sim.applySingleQubitGateForTest(xGate, {}, 0);
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(
                  "failed to apply resident Metal single-qubit gate"),
              std::string::npos)
        << error.what();
    EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
  } catch (...) {
    FAIL() << "expected runtime_error from resident single-gate failure";
  }

  try {
    (void)sim.stateVectorForTest();
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(
                  "unrecoverable Metal resident state"),
              std::string::npos)
        << error.what();
    EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
    return;
  }

  FAIL() << "expected resident single-gate failure to poison readback";
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorPoisonsResidentStateWhenTwoGateFails) {
  const std::vector<std::complex<double>> xGate{{0.0, 0.0},
                                                {1.0, 0.0},
                                                {1.0, 0.0},
                                                {0.0, 0.0}};
  const std::vector<std::complex<double>> swapGate{
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
      {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
      {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};

  MklqMetalCircuitSimulatorTester sim;
  if (!sim.metalRuntimeAvailableForTest())
    return;

  sim.setStateForTest(
      {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}});
  sim.applySingleQubitGateForTest(xGate, {}, 0);
  ASSERT_EQ(sim.residentStateUploadsForTest(), 1);
  ASSERT_EQ(sim.singleQubitApplicationsForTest(), 1);
  ASSERT_EQ(sim.residentStateDownloadsForTest(), 0);
  sim.setResidentFailureModeForTest(ResidentFailureMode::TwoGate);

  try {
    sim.applyGateTaskForTest("swap", swapGate, {}, {0, 1});
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(
                  "failed to apply resident Metal two-qubit gate"),
              std::string::npos)
        << error.what();
    EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
  } catch (...) {
    FAIL() << "expected runtime_error from resident two-gate failure";
  }

  try {
    (void)sim.stateVectorForTest();
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(
                  "unrecoverable Metal resident state"),
              std::string::npos)
        << error.what();
    EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
    return;
  }

  FAIL() << "expected resident two-gate failure to poison readback";
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorThrowsWhenResidentMeasurementProbabilityFails) {
  const std::vector<std::complex<double>> xGate{{0.0, 0.0},
                                                {1.0, 0.0},
                                                {1.0, 0.0},
                                                {0.0, 0.0}};

  MklqMetalCircuitSimulatorTester sim;
  if (!sim.metalRuntimeAvailableForTest())
    return;

  sim.setStateForTest({{1.0, 0.0}, {0.0, 0.0}});
  sim.applySingleQubitGateForTest(xGate, {}, 0);
  ASSERT_EQ(sim.residentStateUploadsForTest(), 1);
  ASSERT_EQ(sim.singleQubitApplicationsForTest(), 1);
  ASSERT_EQ(sim.residentStateDownloadsForTest(), 0);
  sim.setResidentFailureModeForTest(ResidentFailureMode::Probability);

  bool threwProbabilityFailure = false;
  try {
    (void)sim.measureQubitForTest(0);
  } catch (const std::runtime_error &error) {
    threwProbabilityFailure = true;
    EXPECT_NE(std::string(error.what()).find(
                  "failed to compute Metal resident measurement probability"),
              std::string::npos)
        << error.what();
    EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
  }

  ASSERT_TRUE(threwProbabilityFailure)
      << "expected resident measurement probability failure to throw";
  const auto state = sim.stateVectorForTest();
  ASSERT_EQ(state.size(), 2);
  expectNear(state[0], {0.0, 0.0});
  expectNear(state[1], {1.0, 0.0});
  EXPECT_EQ(sim.residentStateDownloadsForTest(), 1);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorThrowsWhenResidentMeasurementCollapseFails) {
  const std::vector<std::complex<double>> xGate{{0.0, 0.0},
                                                {1.0, 0.0},
                                                {1.0, 0.0},
                                                {0.0, 0.0}};

  MklqMetalCircuitSimulatorTester sim;
  if (!sim.metalRuntimeAvailableForTest())
    return;

  sim.setStateForTest({{1.0, 0.0}, {0.0, 0.0}});
  sim.applySingleQubitGateForTest(xGate, {}, 0);
  ASSERT_EQ(sim.residentStateUploadsForTest(), 1);
  ASSERT_EQ(sim.singleQubitApplicationsForTest(), 1);
  ASSERT_EQ(sim.residentStateDownloadsForTest(), 0);
  sim.setResidentFailureModeForTest(ResidentFailureMode::Collapse);

  try {
    (void)sim.measureQubitForTest(0);
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(
                  "failed to collapse Metal resident measurement branch"),
              std::string::npos)
        << error.what();
    EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
  } catch (...) {
    FAIL() << "expected runtime_error from resident collapse failure";
  }

  try {
    (void)sim.stateVectorForTest();
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(
                  "unrecoverable Metal resident state"),
              std::string::npos)
        << error.what();
    EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
    return;
  }

  FAIL() << "expected resident collapse failure to poison readback";
}

CUDAQ_TEST(MKLQMetalTester, SimulatorThrowsWhenResidentResetGateFails) {
  const std::vector<std::complex<double>> xGate{{0.0, 0.0},
                                                {1.0, 0.0},
                                                {1.0, 0.0},
                                                {0.0, 0.0}};

  MklqMetalCircuitSimulatorTester sim;
  if (!sim.metalRuntimeAvailableForTest())
    return;

  sim.setStateForTest({{1.0, 0.0}, {0.0, 0.0}});
  sim.applySingleQubitGateForTest(xGate, {}, 0);
  ASSERT_EQ(sim.residentStateUploadsForTest(), 1);
  ASSERT_EQ(sim.singleQubitApplicationsForTest(), 1);
  ASSERT_EQ(sim.residentStateDownloadsForTest(), 0);
  sim.setResidentFailureModeForTest(ResidentFailureMode::Reset);

  try {
    sim.resetQubit(0);
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(
                  "failed to reset Metal resident qubit"),
              std::string::npos)
        << error.what();
    EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
  } catch (...) {
    FAIL() << "expected runtime_error from resident reset failure";
  }

  try {
    (void)sim.stateVectorForTest();
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(
                  "unrecoverable Metal resident state"),
              std::string::npos)
        << error.what();
    EXPECT_EQ(sim.residentStateDownloadsForTest(), 0);
    return;
  }

  FAIL() << "expected resident reset failure to poison readback";
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorSynchronizesResidentStateBeforeZeroShotExpectation) {
  const std::vector<std::complex<double>> xGate{{0.0, 0.0},
                                                {1.0, 0.0},
                                                {1.0, 0.0},
                                                {0.0, 0.0}};

  MklqMetalCircuitSimulatorTester sim;
  sim.setStateForTest({{1.0, 0.0}, {0.0, 0.0}});
  sim.applySingleQubitGateForTest(xGate, {}, 0);

  const auto counts = sim.sampleQubitsForTest({0}, 0);

  ASSERT_TRUE(counts.expectationValue.has_value());
  EXPECT_NEAR(*counts.expectationValue, -1.0, 1.0e-12);
  EXPECT_EQ(sim.residentStateDownloadsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
}

CUDAQ_TEST(MKLQMetalTester,
           SimulatorSamplesDenseFullRegisterThroughMetalProbabilityFill) {
  constexpr std::size_t qubitCount = 7;
  constexpr std::size_t dimension = 1ULL << qubitCount;
  const double amplitude = 1.0 / std::sqrt(static_cast<double>(dimension));

  std::vector<std::complex<double>> state(dimension, {amplitude, 0.0});

  MklqMetalCircuitSimulatorTester sim;
  sim.setStateForTest(std::move(state));

  std::vector<std::size_t> qubits;
  qubits.reserve(qubitCount);
  for (std::size_t qubit = 0; qubit < qubitCount; ++qubit)
    qubits.push_back(qubit);

  const auto counts = sim.sampleQubitsForTest(qubits, 8);
  std::size_t totalShots = 0;
  for (const auto &[bits, count] : counts.counts)
    totalShots += count;

  EXPECT_EQ(totalShots, 8);
  EXPECT_EQ(sim.probabilityFillApplicationsForTest(),
            sim.metalRuntimeAvailableForTest() ? 1 : 0);
}
