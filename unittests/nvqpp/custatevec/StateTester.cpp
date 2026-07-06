/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecCircuitSimulatorEx.h"
#include "CuStateVecError.h"
#include "CuStateVecSimulationState.h"
#include "CuStateVecState.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <sstream>
#include <type_traits>
#include <vector>

namespace {

class HostStateImportSimulator
    : public cudaq::cusv::CuStateVecCircuitSimulator<double> {
public:
  const std::vector<std::complex<double>> readStateVector() {
    return readState();
  }

  bool stagedHostState = false;

protected:
  void writeState(const std::vector<std::complex<double>> &values) override {
    stagedHostState = true;
    CuStateVecCircuitSimulator<double>::writeState(values);
  }
};

} // namespace

TEST(CuStateVecCircuitSimulator, ImportsFullHostStateWithoutStaging) {
  using Complex = std::complex<double>;
  const std::vector<Complex> expected{{0.0, 0.0}, {1.0, 0.0}};
  HostStateImportSimulator simulator;

  simulator.allocateQubits(1, expected.data(),
                           cudaq::simulation_precision::fp64);

  EXPECT_FALSE(simulator.stagedHostState);
  EXPECT_EQ(expected, simulator.readStateVector());
}

TEST(CuStateVecState, ImportsDevicePointerWithoutHostStaging) {
  using Complex = std::complex<double>;
  const std::vector<Complex> expected{{0.0, 0.0}, {1.0, 0.0}};
  Complex *deviceData = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&deviceData),
                                    expected.size() * sizeof(Complex)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(deviceData, expected.data(),
                                    expected.size() * sizeof(Complex),
                                    cudaMemcpyHostToDevice));

  int32_t device = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
  auto state = cudaq::cusv::CuStateVecState<double>::createSingleDevice(
      1, 1, device, false);
  state.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, 1);
  EXPECT_TRUE(state.setStateFromDevicePointer(deviceData, expected.size()));

  std::vector<Complex> actual(expected.size());
  state.getState(actual.data(), 0, actual.size());
  state.synchronize();
  EXPECT_EQ(expected, actual);
  EXPECT_EQ(cudaSuccess, cudaFree(deviceData));
}

TEST(CuStateVecState, UsesDedicatedNonBlockingStream) {
  int32_t device = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
  auto state = cudaq::cusv::CuStateVecState<double>::createSingleDevice(
      1, 1, device, false);
  state.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, 1);
  const auto indices = state.deviceSubStateIndices();
  ASSERT_EQ(indices.size(), 1u);
  const auto resource = state.deviceSubStateVector(indices.front());
  ASSERT_NE(resource.stream, nullptr);
  unsigned flags = 0;
  ASSERT_EQ(cudaSuccess, cudaStreamGetFlags(resource.stream, &flags));
  EXPECT_EQ(flags & cudaStreamNonBlocking,
            static_cast<unsigned>(cudaStreamNonBlocking));
}

TEST(CuStateVecState, AppendsCustomStateWithoutHostStateStaging) {
  using Complex = std::complex<double>;
  int32_t device = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
  auto state = cudaq::cusv::CuStateVecState<double>::createSingleDevice(
      2, 2, device, false);
  state.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, 1);
  const std::vector<Complex> initial{{0.0, 0.0}, {1.0, 0.0}};
  state.setState(initial.data(), 0, initial.size());
  state.synchronize();

  const double scale = std::sqrt(0.5);
  const std::vector<Complex> added{{scale, 0.0}, {scale, 0.0}};
  Complex *deviceAdded = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&deviceAdded),
                                    added.size() * sizeof(Complex)));
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(deviceAdded, added.data(),
                       added.size() * sizeof(Complex), cudaMemcpyHostToDevice));

  EXPECT_TRUE(state.appendState(deviceAdded, added.size()));
  std::vector<Complex> actual(4);
  state.getState(actual.data(), 0, actual.size());
  state.synchronize();
  const std::vector<Complex> expected{
      {0.0, 0.0}, {scale, 0.0}, {0.0, 0.0}, {scale, 0.0}};
  EXPECT_EQ(expected, actual);
  EXPECT_EQ(cudaSuccess, cudaFree(deviceAdded));
}

TEST(CuStateVecState, AppendsHostStateWithDeviceKroneckerProduct) {
  using Complex = std::complex<double>;
  int32_t device = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
  auto state = cudaq::cusv::CuStateVecState<double>::createSingleDevice(
      2, 2, device, false);
  state.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, 1);
  const std::vector<Complex> initial{{0.0, 0.0}, {1.0, 0.0}};
  state.setState(initial.data(), 0, initial.size());
  state.synchronize();

  const double scale = std::sqrt(0.5);
  const std::vector<Complex> added{{scale, 0.0}, {scale, 0.0}};
  EXPECT_TRUE(state.appendState(added.data(), added.size()));
  std::vector<Complex> actual(4);
  state.getState(actual.data(), 0, actual.size());
  state.synchronize();
  const std::vector<Complex> expected{
      {0.0, 0.0}, {scale, 0.0}, {0.0, 0.0}, {scale, 0.0}};
  EXPECT_EQ(expected, actual);
}

TEST(CuStateVecState, ExSamplingOrdersUnsortedRandomInputsEquivalently) {
  using Complex = std::complex<double>;
  int32_t device = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
  auto state = cudaq::cusv::CuStateVecState<double>::createSingleDevice(
      2, 2, device, false);
  state.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, 2);
  const std::vector<Complex> uniform(4, Complex{0.5, 0.0});
  state.setState(uniform.data(), 0, uniform.size());
  state.synchronize();

  const int32_t wires[] = {0, 1};
  const std::vector<double> unsorted{0.8, 0.1, 0.6, 0.3};
  auto sorted = unsorted;
  std::sort(sorted.begin(), sorted.end());
  std::vector<custatevecIndex_t> unsortedOutput(unsorted.size());
  std::vector<custatevecIndex_t> sortedOutput(sorted.size());
  HANDLE_CUSTATEVEC_ERROR(custatevecExSample(
      state.descriptor(), unsortedOutput.data(), wires, 2, unsorted.data(),
      unsorted.size(), CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER, nullptr));
  HANDLE_CUSTATEVEC_ERROR(custatevecExSample(
      state.descriptor(), sortedOutput.data(), wires, 2, sorted.data(),
      sorted.size(), CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER, nullptr));
  EXPECT_EQ(unsortedOutput, sortedOutput);
  EXPECT_TRUE(std::is_sorted(unsortedOutput.begin(), unsortedOutput.end()));
}

template <typename Scalar>
class CuStateVecMigratedCopyTester : public ::testing::Test {};

using MigratedCopyScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CuStateVecMigratedCopyTester, MigratedCopyScalarTypes);

TYPED_TEST(CuStateVecMigratedCopyTester, CopiesWithoutStagingHostSubStates) {
  using Scalar = TypeParam;
  using Complex = std::complex<Scalar>;
  int32_t device = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
  auto source = cudaq::cusv::CuStateVecState<Scalar>::createSingleDevice(
      4, 2, device, false);
  auto destination = cudaq::cusv::CuStateVecState<Scalar>::createSingleDevice(
      4, 2, device, false);
  for (auto *state : {&source, &destination}) {
    state->addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, 2);
    state->addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_MIGRATION, 2);
  }

  std::vector<Complex> first(16);
  for (std::size_t index = 0; index < first.size(); ++index)
    first[index] = {static_cast<Scalar>(index + 1),
                    -static_cast<Scalar>(index)};
  source.setState(first.data(), 0, first.size());
  source.synchronize();
  const std::vector<Complex> destinationValues(
      first.size(), Complex{-static_cast<Scalar>(1), 0});
  destination.setState(destinationValues.data(), 0, destinationValues.size());
  destination.synchronize();
  source.stageSubStateVector(0);
  destination.stageSubStateVector(0);
  EXPECT_EQ(source.deviceSubStateIndices(), (std::vector<int32_t>{0}));
  EXPECT_EQ(destination.deviceSubStateIndices(), (std::vector<int32_t>{0}));

  destination.copyFrom(source);
  EXPECT_EQ(source.deviceSubStateIndices(), (std::vector<int32_t>{0}));
  EXPECT_EQ(destination.deviceSubStateIndices(), (std::vector<int32_t>{0}));
  std::vector<Complex> actual(first.size());
  destination.getState(actual.data(), 0, actual.size());
  destination.synchronize();
  EXPECT_EQ(actual, first);

  std::vector<Complex> second(16);
  for (std::size_t index = 0; index < second.size(); ++index)
    second[index] = {-static_cast<Scalar>(index),
                     static_cast<Scalar>(index + 1)};
  source.setState(second.data(), 0, second.size());
  source.synchronize();
  source.stageSubStateVector(1);
  destination.stageSubStateVector(2);
  EXPECT_EQ(source.deviceSubStateIndices(), (std::vector<int32_t>{1}));
  EXPECT_EQ(destination.deviceSubStateIndices(), (std::vector<int32_t>{2}));

  destination.copyFrom(source);
  EXPECT_EQ(source.deviceSubStateIndices(), (std::vector<int32_t>{1}));
  EXPECT_EQ(destination.deviceSubStateIndices(), (std::vector<int32_t>{2}));
  destination.getState(actual.data(), 0, actual.size());
  destination.synchronize();
  EXPECT_EQ(actual, second);
}

TYPED_TEST(CuStateVecMigratedCopyTester, OverlapPreservesHostDevicePlacement) {
  using Scalar = TypeParam;
  using Complex = std::complex<Scalar>;
  int32_t device = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
  auto left = cudaq::cusv::CuStateVecState<Scalar>::createSingleDevice(
      4, 2, device, false);
  auto right = cudaq::cusv::CuStateVecState<Scalar>::createSingleDevice(
      4, 2, device, false);
  for (auto *state : {&left, &right}) {
    state->addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, 2);
    state->addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_MIGRATION, 2);
  }

  std::vector<Complex> leftValues(16);
  std::vector<Complex> rightValues(16);
  for (std::size_t index = 0; index < leftValues.size(); ++index) {
    leftValues[index] = {static_cast<Scalar>(index + 1),
                         static_cast<Scalar>(index % 3)};
    rightValues[index] = {static_cast<Scalar>(2 * index + 1),
                          -static_cast<Scalar>(index % 5)};
  }
  left.setState(leftValues.data(), 0, leftValues.size());
  right.setState(rightValues.data(), 0, rightValues.size());
  left.synchronize();
  right.synchronize();
  left.stageSubStateVector(1);
  right.stageSubStateVector(2);

  cudaq::cusv::CuStateVecSimulationState<Scalar> leftSimulation(
      std::move(left));
  cudaq::cusv::CuStateVecSimulationState<Scalar> rightSimulation(
      std::move(right));
  std::complex<double> expected{};
  for (std::size_t index = 0; index < leftValues.size(); ++index)
    expected += std::conj(std::complex<double>(leftValues[index])) *
                std::complex<double>(rightValues[index]);
  const double tolerance = std::is_same_v<Scalar, float> ? 1e-4 : 1e-10;
  const auto overlap = leftSimulation.overlap(rightSimulation);
  EXPECT_NEAR(overlap.real(), std::abs(expected), tolerance);
  EXPECT_NEAR(overlap.imag(), 0.0, tolerance);
  EXPECT_EQ(leftSimulation.state().deviceSubStateIndices(),
            (std::vector<int32_t>{1}));
  EXPECT_EQ(rightSimulation.state().deviceSubStateIndices(),
            (std::vector<int32_t>{2}));
}

template <typename Scalar>
class CuStateVecScalarStateTester : public ::testing::Test {};

using ScalarStateTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CuStateVecScalarStateTester, ScalarStateTypes);

TYPED_TEST(CuStateVecScalarStateTester, OwnsZeroQubitDeviceAmplitude) {
  using Scalar = TypeParam;
  using Complex = std::complex<Scalar>;
  const Complex expected{static_cast<Scalar>(0.6), static_cast<Scalar>(0.8)};
  auto simulation = cudaq::cusv::CuStateVecSimulationState<Scalar>::create(
      1, &expected, false);

  EXPECT_EQ(simulation->getNumQubits(), 0u);
  EXPECT_EQ(simulation->getNumElements(), 1u);
  EXPECT_TRUE(simulation->isDeviceData());
  EXPECT_TRUE(simulation->isArrayLike());
  const auto expectedPrecision = std::is_same_v<Scalar, float>
                                     ? cudaq::SimulationState::precision::fp32
                                     : cudaq::SimulationState::precision::fp64;
  EXPECT_EQ(simulation->getPrecision(), expectedPrecision);

  const auto tensor = simulation->getTensor();
  ASSERT_NE(tensor.data, nullptr);
  EXPECT_EQ(tensor.extents, (std::vector<std::size_t>{1}));
  EXPECT_EQ(tensor.fp_precision, expectedPrecision);
  Complex tensorValue;
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(&tensorValue, tensor.data, sizeof(tensorValue),
                       cudaMemcpyDeviceToHost));
  EXPECT_EQ(tensorValue, expected);

  std::vector<Complex> host(1);
  simulation->toHost(host.data(), host.size());
  EXPECT_EQ(host.front(), expected);
  EXPECT_EQ((*simulation)(0, {0}), std::complex<double>(expected));
  EXPECT_EQ(simulation->getAmplitude({}), std::complex<double>(expected));

  auto other = cudaq::cusv::CuStateVecSimulationState<Scalar>::create(
      1, &expected, false);
  EXPECT_THROW(simulation->overlap(*other), std::invalid_argument);

  simulation->destroyState();
  EXPECT_EQ(simulation->getNumElements(), 0u);
  EXPECT_THROW(simulation->getPrecision(), std::runtime_error);
}

TYPED_TEST(CuStateVecScalarStateTester, CopiesDeviceInput) {
  using Scalar = TypeParam;
  using Complex = std::complex<Scalar>;
  const Complex expected{static_cast<Scalar>(-0.25), static_cast<Scalar>(0.75)};
  Complex *deviceInput = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&deviceInput),
                                    sizeof(Complex)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(deviceInput, &expected, sizeof(Complex),
                                    cudaMemcpyHostToDevice));

  auto simulation = cudaq::cusv::CuStateVecSimulationState<Scalar>::create(
      1, deviceInput, false);
  ASSERT_EQ(cudaSuccess, cudaFree(deviceInput));
  std::vector<Complex> host(1);
  simulation->toHost(host.data(), host.size());
  EXPECT_EQ(host.front(), expected);
}

TEST(CuStateVecSimulationState, MatchesLegacyDumpAndDestroyedPrecision) {
  using Complex = std::complex<double>;
  int32_t device = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
  auto state = cudaq::cusv::CuStateVecState<double>::createSingleDevice(
      1, 1, device, false);
  state.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, 1);
  const std::vector<Complex> values{{0.0, 0.0}, {1.0, 0.0}};
  state.setState(values.data(), 0, values.size());
  state.synchronize();

  cudaq::cusv::CuStateVecSimulationState<double> simulation(std::move(state));
  std::ostringstream output;
  simulation.dump(output);
  EXPECT_EQ(output.str(), "SV: [(0,0), (1,0)]\n");
  simulation.destroyState();
  EXPECT_THROW(simulation.getPrecision(), std::runtime_error);

  std::ostringstream destroyedOutput;
  simulation.dump(destroyedOutput);
  EXPECT_EQ(destroyedOutput.str(), "SV: nullptr\n");
}

TEST(CuStateVecSimulationState, MigratedStorageIsNotArrayLike) {
  int32_t device = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
  auto state = cudaq::cusv::CuStateVecState<double>::createSingleDevice(
      2, 1, device, false);
  state.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, 1);
  state.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_MIGRATION, 1);
  cudaq::cusv::CuStateVecSimulationState<double> simulation(std::move(state));
  EXPECT_FALSE(simulation.isArrayLike());
  EXPECT_NEAR(std::abs(simulation.getAmplitude({0, 0}) - 1.0), 0.0, 1e-12);
}
