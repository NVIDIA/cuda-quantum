/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatOpConverter.h"
#include "CuDensityMatState.h"
#include "CuDensityMatTimeStepper.h"
#include "cudaq/operators.h"
#include <CuDensityMatErrorHandling.h>
#include <complex>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace cudaq;

namespace {
using Operator = sum_op<matrix_handler>;
using Parameters = std::unordered_map<std::string, std::complex<double>>;

constexpr const char *maxFusedDimensionVar =
    "CUDAQ_DYNAMICS_MAX_FUSED_DIMENSION";

// Creates a converter with fusion capped at `maxFusedDimension`; "0" disables
// fusion. The cap is read when the converter is constructed.
std::unique_ptr<dynamics::CuDensityMatOpConverter>
makeConverter(cudensitymatHandle_t handle, const char *maxFusedDimension) {
  const char *previous = std::getenv(maxFusedDimensionVar);
  const std::optional<std::string> saved =
      previous ? std::optional<std::string>(previous) : std::nullopt;
  setenv(maxFusedDimensionVar, maxFusedDimension, 1);
  auto converter = std::make_unique<dynamics::CuDensityMatOpConverter>(handle);
  if (saved)
    setenv(maxFusedDimensionVar, saved->c_str(), 1);
  else
    unsetenv(maxFusedDimensionVar);
  return converter;
}

std::size_t hilbertSpaceDimension(const std::vector<int64_t> &dims) {
  std::size_t dim = 1;
  for (auto extent : dims)
    dim *= static_cast<std::size_t>(extent);
  return dim;
}

// Fixed pseudo-random values for a state. The Liouvillian is linear, so the
// state need not be normalized or a physical density matrix. Density matrices
// are stored column-major.
std::vector<std::complex<double>> randomValues(std::size_t count) {
  std::mt19937 generator(1234);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  std::vector<std::complex<double>> data(count);
  for (auto &value : data)
    value = {distribution(generator), distribution(generator)};
  return data;
}

// The matrix of `op` on the full space, with the first degree varying fastest
// as in the state.
complex_matrix fullMatrix(const Operator &op, const std::vector<int64_t> &dims,
                          const Parameters &parameters) {
  dimension_map dimensions;
  for (std::size_t i = 0; i < dims.size(); ++i)
    dimensions[i] = dims[i];
  const auto degrees = op.degrees();
  const auto subMatrix = op.to_matrix(dimensions, parameters);
  const auto dim = hilbertSpaceDimension(dims);
  std::vector<bool> isOpDegree(dims.size(), false);
  for (auto degree : degrees)
    isOpDegree[degree] = true;
  // Splits a full index into the index on the operator's degrees and the
  // index on the remaining degrees.
  const auto split = [&](std::size_t index) {
    std::size_t opIndex = 0, opStride = 1, restIndex = 0, restStride = 1;
    for (std::size_t degree = 0; degree < dims.size(); ++degree) {
      const auto extent = static_cast<std::size_t>(dims[degree]);
      const auto digit = index % extent;
      index /= extent;
      if (isOpDegree[degree]) {
        opIndex += digit * opStride;
        opStride *= extent;
      } else {
        restIndex += digit * restStride;
        restStride *= extent;
      }
    }
    return std::make_pair(opIndex, restIndex);
  };
  complex_matrix result(dim, dim);
  for (std::size_t col = 0; col < dim; ++col) {
    const auto [opCol, restCol] = split(col);
    for (std::size_t row = 0; row < dim; ++row) {
      const auto [opRow, restRow] = split(row);
      if (restRow == restCol)
        result[{row, col}] = subMatrix[{opRow, opCol}];
    }
  }
  return result;
}

// -i (H rho - rho H^dagger) + sum_k (L_k rho L_k^dagger
//   - 1/2 (L_k^dagger L_k rho + rho L_k^dagger L_k)), computed on the host.
std::vector<std::complex<double>> hostLiouvillianAction(
    const Operator &ham, const std::vector<Operator> &collapseOps,
    const std::vector<int64_t> &dims, const Parameters &parameters,
    const std::vector<std::complex<double>> &rhoData) {
  const auto dim = hilbertSpaceDimension(dims);
  complex_matrix rho(dim, dim);
  for (std::size_t col = 0; col < dim; ++col)
    for (std::size_t row = 0; row < dim; ++row)
      rho[{row, col}] = rhoData[row + dim * col];

  auto h = fullMatrix(ham, dims, parameters);
  const auto hDag = h.adjoint();
  complex_matrix result = std::complex<double>(0.0, -1.0) * (h * rho) +
                          std::complex<double>(0.0, 1.0) * (rho * hDag);
  for (const auto &collapseOp : collapseOps) {
    auto l = fullMatrix(collapseOp, dims, parameters);
    const auto lDag = l.adjoint();
    const auto lDagL = lDag * l;
    result += l * rho * lDag;
    result += std::complex<double>(-0.5, 0.0) * (lDagL * rho);
    result += std::complex<double>(-0.5, 0.0) * (rho * lDagL);
  }

  std::vector<std::complex<double>> resultData(dim * dim);
  for (std::size_t col = 0; col < dim; ++col)
    for (std::size_t row = 0; row < dim; ++row)
      resultData[row + dim * col] = result[{row, col}];
  return resultData;
}

// -i H psi, computed on the host.
std::vector<std::complex<double>>
hostHamiltonianAction(const Operator &ham, const std::vector<int64_t> &dims,
                      const Parameters &parameters,
                      const std::vector<std::complex<double>> &psi) {
  const auto h = fullMatrix(ham, dims, parameters);
  std::vector<std::complex<double>> result(psi.size());
  for (std::size_t row = 0; row < psi.size(); ++row)
    for (std::size_t col = 0; col < psi.size(); ++col)
      result[row] += std::complex<double>(0.0, -1.0) * h[{row, col}] * psi[col];
  return result;
}

void expectAllNear(const std::vector<std::complex<double>> &actual,
                   const std::vector<std::complex<double>> &expected,
                   const std::string &label) {
  ASSERT_EQ(actual.size(), expected.size());
  double maxDiff = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i)
    maxDiff = std::max(maxDiff, std::abs(actual[i] - expected[i]));
  EXPECT_LT(maxDiff, 1e-12) << label;
}
} // namespace

class OperatorFusionTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle_;

  void SetUp() override { HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_)); }

  void TearDown() override { HANDLE_CUDM_ERROR(cudensitymatDestroy(handle_)); }

  // Applies the Liouvillian built with fusion capped at `maxFusedDimension`
  // to `stateData`, a density matrix for the master equation and a state
  // vector otherwise.
  std::vector<std::complex<double>>
  applyLiouvillian(const char *maxFusedDimension, const Operator &ham,
                   const std::vector<Operator> &collapseOps,
                   const std::vector<int64_t> &dims,
                   const Parameters &parameters,
                   const std::vector<std::complex<double>> &stateData,
                   bool isMasterEquation = true) {
    auto converter = makeConverter(handle_, maxFusedDimension);
    auto liouvillian = converter->constructLiouvillian(
        {ham}, {collapseOps}, dims, parameters, isMasterEquation);
    auto inputState = cudaq::state::from_data(stateData);
    auto *simState = dynamic_cast<CuDensityMatState *>(
        cudaq::state_helper::getSimulationState(&inputState));
    EXPECT_NE(simState, nullptr);
    simState->initialize_cudm(handle_, dims, /*batchSize=*/1);
    CuDensityMatTimeStepper stepper(handle_, liouvillian);
    auto outputState = stepper.compute(inputState, 0.0, parameters);
    std::vector<std::complex<double>> result(stateData.size());
    outputState.to_host(result.data(), result.size());
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(liouvillian));
    return result;
  }

  // Caps that change which terms fuse and how large the superoperator
  // windows grow. "0" disables fusion.
  static constexpr const char *maxFusedDimensions[] = {"0", "4", "16", "64",
                                                       "256"};

  // Checks the master equation Liouvillian against the host reference for
  // each fusion cap.
  void checkFusionMatches(const Operator &ham,
                          const std::vector<Operator> &collapseOps,
                          const std::vector<int64_t> &dims,
                          const Parameters &parameters = {}) {
    const auto dim = hilbertSpaceDimension(dims);
    const auto rho = randomValues(dim * dim);
    const auto expected =
        hostLiouvillianAction(ham, collapseOps, dims, parameters, rho);
    for (const char *maxFusedDimension : maxFusedDimensions)
      expectAllNear(applyLiouvillian(maxFusedDimension, ham, collapseOps, dims,
                                     parameters, rho),
                    expected,
                    std::string("CUDAQ_DYNAMICS_MAX_FUSED_DIMENSION=") +
                        maxFusedDimension);
  }

  // Checks the state vector Liouvillian, -i H, against the host reference for
  // each fusion cap.
  void checkStateVectorFusionMatches(const Operator &ham,
                                     const std::vector<int64_t> &dims) {
    const auto psi = randomValues(hilbertSpaceDimension(dims));
    const auto expected = hostHamiltonianAction(ham, dims, {}, psi);
    for (const char *maxFusedDimension : maxFusedDimensions)
      expectAllNear(applyLiouvillian(maxFusedDimension, ham, {}, dims, {}, psi,
                                     /*isMasterEquation=*/false),
                    expected,
                    std::string("CUDAQ_DYNAMICS_MAX_FUSED_DIMENSION=") +
                        maxFusedDimension);
  }
};

// Bond terms and single-site fields fuse into windowed superoperators, along
// with the single-site decay operators. The complex coupling makes the
// Hamiltonian non-Hermitian, so its left and right actions differ.
TEST_F(OperatorFusionTest, HeisenbergChainWithDecay) {
  const std::vector<int64_t> dims(4, 2);
  auto spinHam = spin_op::empty();
  for (std::size_t i = 0; i + 1 < dims.size(); ++i) {
    spinHam += 1.0 * spin_op::x(i) * spin_op::x(i + 1);
    spinHam += 0.8 * spin_op::y(i) * spin_op::y(i + 1);
    spinHam += 0.6 * spin_op::z(i) * spin_op::z(i + 1);
  }
  for (std::size_t i = 0; i < dims.size(); ++i)
    spinHam += (0.1 * (i + 1)) * spin_op::z(i);
  spinHam += std::complex<double>(0.05, 0.1) * spin_op::x(0) * spin_op::y(1);
  std::vector<Operator> collapseOps;
  for (std::size_t i = 0; i < dims.size(); ++i)
    collapseOps.emplace_back(0.3 * boson_op::annihilate(i));
  checkFusionMatches(Operator(spinHam), collapseOps, dims);
}

// The 3-level mode's terms and its non-normal, complex, multi-term collapse
// operator fit in superoperators. The 10-level mode's terms are too large for
// that and fuse on one side only. The complex drive makes the Hamiltonian
// non-Hermitian.
TEST_F(OperatorFusionTest, NonNormalCollapseOperatorsOnMixedModeSizes) {
  const std::vector<int64_t> dims = {3, 10};
  Operator ham(0.7 * boson_op::number(0) + 0.4 * boson_op::number(1) +
               0.25 * (boson_op::create(0) * boson_op::annihilate(1) +
                       boson_op::annihilate(0) * boson_op::create(1)));
  ham += Operator(std::complex<double>(0.1, 0.2) * boson_op::create(0));
  std::vector<Operator> collapseOps = {
      Operator(0.5 * boson_op::annihilate(0) +
               std::complex<double>(0.0, 0.3) * boson_op::number(0)),
      Operator(0.2 * boson_op::annihilate(1))};
  checkFusionMatches(ham, collapseOps, dims);
}

// Hamiltonian terms and collapse operators with a parameterized coefficient
// are not fused and are combined with the fused terms. The parameterized decay
// of the 6-level mode keeps its L^dagger L factors, which fuse into one
// multi-diagonal operator unless fusion is disabled.
TEST_F(OperatorFusionTest, ParameterizedCoefficientsStayUnfused) {
  const std::vector<int64_t> dims = {2, 6};
  const auto omega = [](const Parameters &parameters) {
    return parameters.at("omega");
  };
  const auto gamma = [](const Parameters &parameters) {
    return parameters.at("gamma");
  };
  Operator ham(0.5 * spin_op::z(0));
  ham += Operator(scalar_operator(omega) * spin_op::x(0));
  ham += Operator(0.3 * boson_op::number(1) +
                  0.2 * (boson_op::create(0) * boson_op::annihilate(1) +
                         boson_op::annihilate(0) * boson_op::create(1)));
  std::vector<Operator> collapseOps = {
      Operator(0.4 * boson_op::annihilate(0)),
      Operator(scalar_operator(gamma) * boson_op::annihilate(1))};
  checkFusionMatches(ham, collapseOps, dims,
                     {{"omega", 0.8}, {"gamma", {0.3, 0.1}}});
}

// Collective decay, whose product terms act on different degrees. On two
// qubits it fits in a superoperator. With a 12-level mode only its
// anticommutator part is fused, and L rho L^dagger stays unfused.
TEST_F(OperatorFusionTest, MixedDegreeCollapseOperators) {
  {
    const std::vector<int64_t> dims = {2, 2};
    Operator ham(0.5 * spin_op::z(0) + 0.3 * spin_op::z(1));
    std::vector<Operator> collapseOps = {
        Operator(boson_op::annihilate(0) + boson_op::annihilate(1))};
    checkFusionMatches(ham, collapseOps, dims);
  }
  {
    const std::vector<int64_t> dims = {2, 12};
    Operator ham(0.5 * boson_op::number(0) + 0.3 * boson_op::number(1) +
                 0.2 * (boson_op::create(0) * boson_op::annihilate(1) +
                        boson_op::annihilate(0) * boson_op::create(1)));
    std::vector<Operator> collapseOps = {
        Operator(boson_op::annihilate(0) + 0.5 * boson_op::annihilate(1))};
    checkFusionMatches(ham, collapseOps, dims);
  }
}

// Without collapse operators the state vector is evolved by -i H, whose terms
// fuse on one side.
TEST_F(OperatorFusionTest, StateVectorEvolution) {
  const std::vector<int64_t> dims = {2, 2, 2, 3};
  auto spinHam = spin_op::empty();
  for (std::size_t i = 0; i + 1 < 3; ++i) {
    spinHam += 1.0 * spin_op::x(i) * spin_op::x(i + 1);
    spinHam += 0.7 * spin_op::z(i) * spin_op::z(i + 1);
  }
  for (std::size_t i = 0; i < 3; ++i)
    spinHam += (0.2 * (i + 1)) * spin_op::z(i);
  Operator ham(spinHam);
  ham += Operator(0.4 * boson_op::number(3) +
                  0.3 * (boson_op::create(2) * boson_op::annihilate(3) +
                         boson_op::annihilate(2) * boson_op::create(3)));
  ham += Operator(std::complex<double>(0.1, -0.2) * spin_op::y(0));
  checkStateVectorFusionMatches(ham, dims);
}

TEST_F(OperatorFusionTest, BatchedMixedDegreeCollapseOperatorsThrow) {
  const std::vector<int64_t> dims = {2, 2};
  Operator ham(0.5 * spin_op::z(0));
  Operator collapseOp(boson_op::annihilate(0) + boson_op::annihilate(1));
  for (const char *maxFusedDimension : {"0", "64"}) {
    auto converter = makeConverter(handle_, maxFusedDimension);
    EXPECT_THROW(converter->constructLiouvillian(
                     {ham, ham}, {{collapseOp}, {collapseOp}}, dims, {}, true),
                 std::invalid_argument)
        << "CUDAQ_DYNAMICS_MAX_FUSED_DIMENSION=" << maxFusedDimension;
  }
}
