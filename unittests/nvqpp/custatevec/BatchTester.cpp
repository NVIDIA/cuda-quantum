/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecBatch.h"

#include <gtest/gtest.h>

#include <numeric>

using cudaq::cusv::compactMatrixTask;
using cudaq::cusv::CuStateVecBatch;
using cudaq::cusv::MatrixTask;
using cudaq::cusv::NoiseChannelKind;
using cudaq::cusv::NoiseTask;

namespace {

MatrixTask<double>
matrixTask(std::initializer_list<std::complex<double>> data) {
  MatrixTask<double> task;
  task.matrix = data;
  task.targets = {0};
  return task;
}

} // namespace

TEST(CuStateVecBatchTester, AppliesOneOriginalGatePerBatch) {
  CuStateVecBatch<double> batch(1, 3, false);
  const auto identity = matrixTask({1.0, 0.0, 0.0, 1.0});
  const auto x = matrixTask({0.0, 1.0, 1.0, 0.0});
  batch.apply({identity, x, identity});

  EXPECT_EQ(batch.sample(0, {0}, {0.5}, false).counts.at("0"), 1);
  EXPECT_EQ(batch.sample(1, {0}, {0.5}, false).counts.at("1"), 1);
  EXPECT_EQ(batch.sample(2, {0}, {0.5}, false).counts.at("0"), 1);
}

TEST(CuStateVecBatchTester, CompactsStructuredMatrixTasks) {
  auto diagonal = matrixTask({1.0, 0.0, 0.0, -1.0});
  compactMatrixTask(diagonal);
  EXPECT_EQ(diagonal.matrixType, CUSTATEVEC_EX_MATRIX_DIAGONAL);
  EXPECT_EQ(diagonal.matrix, (std::vector<std::complex<double>>{1.0, -1.0}));

  auto antiDiagonal = matrixTask({0.0, {0.0, -1.0}, {0.0, 1.0}, 0.0});
  compactMatrixTask(antiDiagonal);
  EXPECT_EQ(antiDiagonal.matrixType, CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL);
  EXPECT_EQ(antiDiagonal.matrix,
            (std::vector<std::complex<double>>{{0.0, -1.0}, {0.0, 1.0}}));

  const double scale = std::sqrt(0.5);
  auto dense = matrixTask({scale, scale, scale, -scale});
  const auto original = dense.matrix;
  compactMatrixTask(dense);
  EXPECT_EQ(dense.matrixType, CUSTATEVEC_EX_MATRIX_DENSE);
  EXPECT_EQ(dense.matrix, original);
}

TEST(CuStateVecBatchTester, ReusesCapacityForSmallerActiveBatch) {
  CuStateVecBatch<double> batch(1, 3, false, false);
  const auto x = matrixTask({0.0, 1.0, 1.0, 0.0});
  batch.setZeroState();
  batch.apply(x);
  EXPECT_EQ(batch.measure({0}, {0.5, 0.5, 0.5}),
            (std::vector<custatevecIndex_t>{1, 1, 1}));

  batch.resize(1);
  batch.setZeroState();
  batch.apply(x);
  EXPECT_EQ(batch.measure({0}, {0.5}), (std::vector<custatevecIndex_t>{1}));
  EXPECT_THROW(batch.resize(0), std::invalid_argument);
  EXPECT_THROW(batch.resize(4), std::invalid_argument);
  EXPECT_THROW(batch.measure({0}, {0.5, 0.5}), std::invalid_argument);
  EXPECT_THROW(batch.apply(std::vector{x, x}), std::invalid_argument);
}

TEST(CuStateVecBatchTester, BroadcastsDeviceInitialState) {
  using Complex = std::complex<double>;
  const std::vector<Complex> oneState{0.0, 1.0};
  Complex *deviceState = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&deviceState),
                                    oneState.size() * sizeof(Complex)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(deviceState, oneState.data(),
                                    oneState.size() * sizeof(Complex),
                                    cudaMemcpyHostToDevice));

  CuStateVecBatch<double> batch(1, 4, false, false);
  batch.setState(deviceState);
  EXPECT_EQ(batch.measure({0}, {0.5, 0.5, 0.5, 0.5}),
            (std::vector<custatevecIndex_t>{1, 1, 1, 1}));
  EXPECT_EQ(cudaSuccess, cudaFree(deviceState));
}

TEST(CuStateVecBatchTester, AppliesCompactedDeferredGate) {
  CuStateVecBatch<double> batch(1, 3, false);
  auto x = matrixTask({0.0, 1.0, 1.0, 0.0});
  compactMatrixTask(x);
  ASSERT_EQ(x.matrixType, CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL);
  batch.apply(x);
  EXPECT_EQ(batch.measure({0}, {0.5, 0.5, 0.5}),
            (std::vector<custatevecIndex_t>{1, 1, 1}));
}

TEST(CuStateVecBatchTester, MeasuresWholeBatchWithoutCollapse) {
  CuStateVecBatch<double> batch(1, 2, false);
  const auto x = matrixTask({0.0, 1.0, 1.0, 0.0});
  batch.apply({x, x});
  EXPECT_EQ(batch.measure({0}, {0.25, 0.75}),
            (std::vector<custatevecIndex_t>{1, 1}));
  EXPECT_EQ(batch.measure({0}, {0.75, 0.25}),
            (std::vector<custatevecIndex_t>{1, 1}));
}

TEST(CuStateVecBatchTester, AppliesAndNormalizesGeneralKrausBranches) {
  CuStateVecBatch<double> batch(1, 2, false);
  const auto x = matrixTask({0.0, 1.0, 1.0, 0.0});
  batch.apply({x, x});

  NoiseTask<double> damping;
  damping.kind = NoiseChannelKind::General;
  damping.wires = {0};
  damping.matrices = {
      {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {std::sqrt(0.75), 0.0}},
      {{0.0, 0.0}, {0.5, 0.0}, {0.0, 0.0}, {0.0, 0.0}}};
  damping.matrixTypes.assign(2, CUSTATEVEC_EX_MATRIX_DENSE);
  batch.applyNoise(damping, {0.1, 0.9});

  EXPECT_EQ(batch.measure({0}, {0.5, 0.5}),
            (std::vector<custatevecIndex_t>{1, 0}));
}

TEST(CuStateVecBatchTester, ExpandsCompactMixedUnitaryBranches) {
  CuStateVecBatch<double> batch(1, 2, false);

  NoiseTask<double> bitFlip;
  bitFlip.kind = NoiseChannelKind::MixedUnitary;
  bitFlip.wires = {0};
  bitFlip.matrices = {{{1.0, 0.0}, {1.0, 0.0}}, {{1.0, 0.0}, {1.0, 0.0}}};
  bitFlip.matrixTypes = {CUSTATEVEC_EX_MATRIX_DIAGONAL,
                         CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL};
  bitFlip.probabilities = {0.5, 0.5};
  batch.applyNoise(bitFlip, {0.1, 0.9});

  EXPECT_EQ(batch.measure({0}, {0.5, 0.5}),
            (std::vector<custatevecIndex_t>{0, 1}));
}

TEST(CuStateVecBatchTester, PreservesMixedUnitaryResidualIdentity) {
  CuStateVecBatch<double> batch(1, 2, false);

  NoiseTask<double> bitFlip;
  bitFlip.kind = NoiseChannelKind::MixedUnitary;
  bitFlip.wires = {0};
  bitFlip.matrices = {{{1.0, 0.0}, {1.0, 0.0}}};
  bitFlip.matrixTypes = {CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL};
  bitFlip.probabilities = {0.25};
  batch.applyNoise(bitFlip, {0.1, 0.9});

  // The probability not assigned to an operator is the identity branch, as
  // it is in the cuStateVecEx unitary-channel path.
  EXPECT_EQ(batch.measure({0}, {0.5, 0.5}),
            (std::vector<custatevecIndex_t>{1, 0}));
}

TEST(CuStateVecBatchTester, ExpandsCompactGeneralKrausBranches) {
  CuStateVecBatch<double> batch(1, 2, false);
  const auto x = matrixTask({0.0, 1.0, 1.0, 0.0});
  batch.apply({x, x});

  NoiseTask<double> damping;
  damping.kind = NoiseChannelKind::General;
  damping.wires = {0};
  damping.matrices = {
      {{1.0, 0.0}, {std::sqrt(0.75), 0.0}},
      {{0.5, 0.0}, {0.0, 0.0}},
  };
  damping.matrixTypes = {CUSTATEVEC_EX_MATRIX_DIAGONAL,
                         CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL};
  batch.applyNoise(damping, {0.1, 0.9});

  EXPECT_EQ(batch.measure({0}, {0.5, 0.5}),
            (std::vector<custatevecIndex_t>{1, 0}));
}

TEST(CuStateVecBatchTester, ComputesWidePauliTermsWithoutDenseMatrices) {
  constexpr int32_t numWires = 16;
  CuStateVecBatch<double> batch(numWires, 2, false);
  std::vector<custatevecPauli_t> paulis(numWires, CUSTATEVEC_PAULI_Z);
  std::vector<int32_t> targets(numWires);
  std::iota(targets.begin(), targets.end(), 0);
  const auto values = batch.expectationPauli({paulis}, {targets});
  ASSERT_EQ(values.size(), 2);
  EXPECT_NEAR(values[0], 1.0, 1.e-12);
  EXPECT_NEAR(values[1], 1.0, 1.e-12);
}

TEST(CuStateVecBatchTester, RejectsInvalidPauliTerms) {
  CuStateVecBatch<double> batch(2, 2, false);
  EXPECT_THROW(batch.expectationPauli(
                   {{CUSTATEVEC_PAULI_X, CUSTATEVEC_PAULI_Y}}, {{0, 0}}),
               std::invalid_argument);
  EXPECT_THROW(batch.expectationPauli({{CUSTATEVEC_PAULI_Z}}, {{2}}),
               std::invalid_argument);
}

template <typename Scalar>
class CuStateVecBatchExpectationTester : public ::testing::Test {};

using ExpectationScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CuStateVecBatchExpectationTester, ExpectationScalarTypes);

TYPED_TEST(CuStateVecBatchExpectationTester,
           ComputesPauliExpectationsByGateApplication) {
  using Scalar = TypeParam;
  using Complex = std::complex<Scalar>;
  constexpr std::size_t batchSize = 4;
  constexpr int32_t numWires = 9;
  CuStateVecBatch<Scalar> batch(numWires, batchSize, false);
  const Scalar scale = std::sqrt(Scalar{0.5});

  const auto task = [](std::vector<Complex> matrix, int32_t target) {
    MatrixTask<Scalar> result;
    result.matrix = std::move(matrix);
    result.targets = {target};
    return result;
  };
  const auto identity = task({1.0, 0.0, 0.0, 1.0}, 0);
  const auto x = task({0.0, 1.0, 1.0, 0.0}, 0);
  const auto h = task({scale, scale, scale, -scale}, 0);
  const auto yEigenstate =
      task({scale, scale, Complex{0.0, scale}, Complex{0.0, -scale}}, 0);
  batch.apply({identity, x, h, yEigenstate});

  const auto identityOnOne = task({1.0, 0.0, 0.0, 1.0}, 1);
  const auto hOnOne = task({scale, scale, scale, -scale}, 1);
  batch.apply({identityOnOne, identityOnOne, hOnOne, hOnOne});

  std::vector<custatevecPauli_t> widePaulis(numWires, CUSTATEVEC_PAULI_Z);
  std::vector<int32_t> wideTargets(numWires);
  std::iota(wideTargets.begin(), wideTargets.end(), 0);
  const std::vector<std::vector<custatevecPauli_t>> paulis{
      {},
      {CUSTATEVEC_PAULI_Z},
      {CUSTATEVEC_PAULI_X},
      {CUSTATEVEC_PAULI_Y},
      {CUSTATEVEC_PAULI_X, CUSTATEVEC_PAULI_X},
      {CUSTATEVEC_PAULI_Y, CUSTATEVEC_PAULI_X},
      widePaulis};
  const std::vector<std::vector<int32_t>> targets{
      {}, {0}, {0}, {0}, {0, 1}, {0, 1}, wideTargets};

  const auto values = batch.expectationPauli(paulis, targets);
  const std::vector<double> expected{
      1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
      1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,  0.0, 1.0, 0.0, 1.0, 0.0};
  ASSERT_EQ(values.size(), expected.size());
  const double tolerance = std::is_same_v<Scalar, float> ? 1.e-5 : 1.e-12;
  for (std::size_t index = 0; index < values.size(); ++index)
    EXPECT_NEAR(values[index], expected[index], tolerance) << index;

  constexpr std::size_t activeBatchSize = 3;
  batch.resize(activeBatchSize);
  batch.setZeroState();
  const auto tail = batch.expectationPauli(paulis, targets);
  ASSERT_EQ(tail.size(), activeBatchSize * paulis.size());
  const std::vector<double> zeroStateExpected{1.0, 1.0, 0.0, 0.0,
                                              0.0, 0.0, 1.0};
  for (std::size_t state = 0; state < activeBatchSize; ++state)
    for (std::size_t term = 0; term < paulis.size(); ++term)
      EXPECT_NEAR(tail[state * paulis.size() + term], zeroStateExpected[term],
                  tolerance);

  const auto repeated = batch.expectationPauli(paulis, targets);
  ASSERT_EQ(repeated.size(), tail.size());
  for (std::size_t index = 0; index < repeated.size(); ++index)
    EXPECT_NEAR(repeated[index], tail[index], tolerance) << index;
}

TEST(CuStateVecBatchTester, UnsortedRandomsHaveEquivalentOrderedOutput) {
  CuStateVecBatch<double> batch(2, 1, false);
  const double scale = std::sqrt(0.5);
  MatrixTask<double> hadamard;
  hadamard.matrix = {scale, scale, scale, -scale};
  hadamard.targets = {0};
  batch.apply({hadamard});
  hadamard.targets = {1};
  batch.apply({hadamard});

  const std::vector<double> unsorted{0.8, 0.1, 0.6, 0.3};
  auto sorted = unsorted;
  std::sort(sorted.begin(), sorted.end());
  const auto unsortedResult = batch.sample(0, {0, 1}, unsorted, true);
  const auto sortedResult = batch.sample(0, {0, 1}, sorted, true);
  EXPECT_EQ(unsortedResult.counts, sortedResult.counts);
  EXPECT_EQ(unsortedResult.sequentialData, sortedResult.sequentialData);
}

TEST(CuStateVecBatchTester, RejectsMismatchedOperands) {
  CuStateVecBatch<double> batch(2, 2, false);
  auto first = matrixTask({1.0, 0.0, 0.0, 1.0});
  auto second = first;
  second.targets = {1};
  EXPECT_THROW(batch.apply({first, second}), std::invalid_argument);
}
