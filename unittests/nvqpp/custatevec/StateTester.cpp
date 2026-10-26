/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecSimulationState.h"
#include "CuStateVecState.h"
#include "common/EigenDense.h"

#include <gtest/gtest.h>
#include <unsupported/Eigen/KroneckerProduct>

#include <complex>
#include <random>
#include <type_traits>
#include <vector>

namespace {

Eigen::Vector2cd randomState(std::mt19937 &generator) {
  std::normal_distribution<double> distribution(0.0, 1.0);
  Eigen::Vector2cd state;
  for (Eigen::Index index = 0; index < state.size(); ++index)
    state(index) = {distribution(generator), distribution(generator)};
  return state.normalized();
}

} // namespace

// Checks that importing device-resident amplitudes preserves the complete
// state-vector contents.
TEST(CuStateVecState, ImportsDevicePointerState) {
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

// Checks that appending a device-resident state produces the correct Kronecker
// product and qubit ordering.
TEST(CuStateVecState, AppendsDeviceStateWithKroneckerProduct) {
  using Complex = std::complex<double>;
  std::mt19937 generator(123);
  for (std::size_t iteration = 0; iteration < 10; ++iteration) {
    const auto initial = randomState(generator);
    const auto added = randomState(generator);

    int32_t device = 0;
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
    auto state = cudaq::cusv::CuStateVecState<double>::createSingleDevice(
        2, 2, device, false);
    state.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, 1);
    state.setState(initial.data(), 0, initial.size());
    state.synchronize();

    Complex *deviceAdded = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&deviceAdded),
                                      added.size() * sizeof(Complex)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(deviceAdded, added.data(),
                                      added.size() * sizeof(Complex),
                                      cudaMemcpyHostToDevice));

    EXPECT_TRUE(state.appendState(deviceAdded, added.size()));
    const Eigen::Vector4cd expected =
        Eigen::kroneckerProduct(added, initial).eval();
    std::vector<Complex> actual(expected.size());
    state.getState(actual.data(), 0, actual.size());
    state.synchronize();
    for (Eigen::Index index = 0; index < expected.size(); ++index)
      EXPECT_NEAR(
          std::abs(actual[static_cast<std::size_t>(index)] - expected(index)),
          0.0, 1e-12);
    EXPECT_EQ(cudaSuccess, cudaFree(deviceAdded));
  }
}

// Checks that appending a host-resident state produces the correct Kronecker
// product and qubit ordering.
TEST(CuStateVecState, AppendsHostStateWithKroneckerProduct) {
  using Complex = std::complex<double>;
  std::mt19937 generator(123);
  for (std::size_t iteration = 0; iteration < 10; ++iteration) {
    const auto initial = randomState(generator);
    const auto added = randomState(generator);

    int32_t device = 0;
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
    auto state = cudaq::cusv::CuStateVecState<double>::createSingleDevice(
        2, 2, device, false);
    state.addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, 1);
    state.setState(initial.data(), 0, initial.size());
    state.synchronize();

    EXPECT_TRUE(state.appendState(added.data(), added.size()));
    const Eigen::Vector4cd expected =
        Eigen::kroneckerProduct(added, initial).eval();
    std::vector<Complex> actual(expected.size());
    state.getState(actual.data(), 0, actual.size());
    state.synchronize();
    for (Eigen::Index index = 0; index < expected.size(); ++index)
      EXPECT_NEAR(
          std::abs(actual[static_cast<std::size_t>(index)] - expected(index)),
          0.0, 1e-12);
  }
}

template <typename Scalar>
class CuStateVecMigratedCopyTester : public ::testing::Test {};

using MigratedCopyScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CuStateVecMigratedCopyTester, MigratedCopyScalarTypes);

// Checks that copying a migrated state preserves its amplitudes and the
// existing source and destination sub-state placement.
TYPED_TEST(CuStateVecMigratedCopyTester,
           CopiesMigratedStateWithoutChangingPlacement) {
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

// Checks that migrated-state overlap is numerically correct regardless of
// which sub-state is initially device-resident.
TYPED_TEST(CuStateVecMigratedCopyTester, ComputesMigratedOverlap) {
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
}

template <typename Scalar>
class CuStateVecScalarStateTester : public ::testing::Test {};

using ScalarStateTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CuStateVecScalarStateTester, ScalarStateTypes);

// Checks device-backed scalar storage, metadata, and amplitude access for a
// zero-qubit state, plus overlap rejection and destruction behavior.
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
  EXPECT_ANY_THROW(simulation->overlap(*other));

  simulation->destroyState();
  EXPECT_EQ(simulation->getNumElements(), 0u);
  EXPECT_ANY_THROW(simulation->getPrecision());
}

// Checks that a zero-qubit state owns its device-input amplitude independently
// of the lifetime of the caller-owned allocation.
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

// Checks that migrated storage is not array-like but still supports logical
// amplitude access.
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
