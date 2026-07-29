// /*******************************************************************************
//  * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "CuDensityMatState.h"
#include "CuDensityMatUtils.h"
#include "common/EigenDense.h"
#include "test_Mocks.h"
#include "cudaq/algorithms/evolve_internal.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/utils/cudaq_utils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

using namespace cudaq;

class MagnusHighOrderIntegratorTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle_;

  void SetUp() override { HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_)); }
  void TearDown() override { HANDLE_CUDM_ERROR(cudensitymatDestroy(handle_)); }

  // Build a single-qubit density-matrix state initialized to |psi><psi| where
  // psi is the given (normalized) state vector.
  cudaq::state
  makeDensityMatrixState(const std::vector<std::complex<double>> &stateVec,
                         const std::vector<int64_t> &dims) {
    auto sv = cudaq::state::from_data(stateVec);
    auto *svCudm = dynamic_cast<CuDensityMatState *>(
        cudaq::state_helper::getSimulationState(&sv));
    EXPECT_NE(svCudm, nullptr);
    svCudm->initialize_cudm(handle_, dims, /*batchSize=*/1);
    auto dm = std::make_unique<CuDensityMatState>(svCudm->to_density_matrix());
    dm->initialize_cudm(handle_, dims, /*batchSize=*/1);
    return cudaq::state(dm.release());
  }

  cudaq::schedule makeSchedule(double tFinal, std::size_t numDataPoints) {
    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0.0, tFinal, numDataPoints))
      steps.emplace_back(t, 0.0);
    return cudaq::schedule(
        steps, {"t"},
        [](const std::string &, const std::complex<double> &v) { return v; });
  }
};

TEST_F(MagnusHighOrderIntegratorTest, Initialization) {
  EXPECT_NO_THROW(cudaq::integrators::magnus_cf4 m1);
  EXPECT_NO_THROW(cudaq::integrators::magnus_cf4 m2(0.01));
  EXPECT_NO_THROW(cudaq::integrators::magnus_cf4 m3(0.01, /*cache=*/8));
  cudaq::integrators::magnus_cf4 m4;
  const auto stats = m4.getStats();
  EXPECT_EQ(stats.cache_hits, 0u);
  EXPECT_EQ(stats.cache_misses, 0u);
  EXPECT_EQ(stats.steps, 0u);
}

// Closed-system Rabi oscillation on a density matrix should track the analytic
// solution and preserve trace, exercising the unitary CF4 fast path.
TEST_F(MagnusHighOrderIntegratorTest, RabiOscillationDensityMatrix) {
  const std::vector<int64_t> dims = {2};
  const double omega = 2.0 * M_PI * 0.1;
  cudaq::sum_op<cudaq::matrix_handler> ham(omega * cudaq::spin_op::x(0));
  SystemDynamics system(dims, ham);

  cudaq::integrators::magnus_cf4 integrator(0.01);
  auto initialState = makeDensityMatrixState({{1.0, 0.0}, {0.0, 0.0}}, dims);
  integrator.setState(initialState, 0.0);
  auto schedule = makeSchedule(10.0, 11);
  cudaq::integrator_helper::init_system_dynamics(integrator, system, schedule);

  const double tFinal = 10.0;
  integrator.integrate(tFinal);
  auto [t, state] = integrator.getState();
  std::vector<std::complex<double>> rho(4);
  state.to_host(rho.data(), rho.size());

  // rho is column-major 2x2: rho[0]=rho00, rho[3]=rho11.
  const double pop0 = rho[0].real();
  const double pop1 = rho[3].real();
  const double expectedPop0 =
      std::cos(omega * tFinal) * std::cos(omega * tFinal);
  EXPECT_NEAR(pop0 + pop1, 1.0, 1e-4) << "Trace should be preserved";
  EXPECT_NEAR(pop0, expectedPop0, 1e-3);

  // Constant Hamiltonian: after the first step the propagator cache should be
  // reused, so hits should dominate misses.
  const auto stats = integrator.getStats();
  EXPECT_GT(stats.steps, 0u);
  EXPECT_GT(stats.cache_hits, stats.cache_misses)
      << "PWC constant Hamiltonian should reuse cached propagators";
}

// magnus_cf4 (fast path) should agree with the general magnus_expansion on a
// closed-system density matrix evolution.
TEST_F(MagnusHighOrderIntegratorTest, MatchesMagnusExpansion) {
  const std::vector<int64_t> dims = {2};
  const double omega = 2.0 * M_PI * 0.15;
  cudaq::sum_op<cudaq::matrix_handler> ham(omega * cudaq::spin_op::x(0));
  SystemDynamics system(dims, ham);
  const double tFinal = 3.0;

  auto runCf4 = [&]() {
    cudaq::integrators::magnus_cf4 integrator(0.005);
    auto st = makeDensityMatrixState({{1.0, 0.0}, {0.0, 0.0}}, dims);
    integrator.setState(st, 0.0);
    auto sched = makeSchedule(tFinal, 31);
    cudaq::integrator_helper::init_system_dynamics(integrator, system, sched);
    integrator.integrate(tFinal);
    auto [t, state] = integrator.getState();
    std::vector<std::complex<double>> rho(4);
    state.to_host(rho.data(), rho.size());
    return rho;
  };

  auto runMagnus = [&]() {
    cudaq::integrators::magnus_expansion integrator(12, 0.005);
    auto st = makeDensityMatrixState({{1.0, 0.0}, {0.0, 0.0}}, dims);
    integrator.setState(st, 0.0);
    auto sched = makeSchedule(tFinal, 31);
    cudaq::integrator_helper::init_system_dynamics(integrator, system, sched);
    integrator.integrate(tFinal);
    auto [t, state] = integrator.getState();
    std::vector<std::complex<double>> rho(4);
    state.to_host(rho.data(), rho.size());
    return rho;
  };

  auto cf4 = runCf4();
  auto mag = runMagnus();
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(cf4[i].real(), mag[i].real(), 1e-4)
        << "Real mismatch at element " << i;
    EXPECT_NEAR(cf4[i].imag(), mag[i].imag(), 1e-4)
        << "Imag mismatch at element " << i;
  }
}

TEST_F(MagnusHighOrderIntegratorTest, CloneReproducesTrajectory) {
  const std::vector<int64_t> dims = {2};
  cudaq::sum_op<cudaq::matrix_handler> ham(2.0 * M_PI * 0.1 *
                                           cudaq::spin_op::x(0));
  SystemDynamics system(dims, ham);

  cudaq::integrators::magnus_cf4 integrator(0.01);
  auto initialState = makeDensityMatrixState({{1.0, 0.0}, {0.0, 0.0}}, dims);
  integrator.setState(initialState, 0.0);
  auto schedule = makeSchedule(1.0, 11);
  cudaq::integrator_helper::init_system_dynamics(integrator, system, schedule);
  integrator.integrate(0.5);

  auto cloned = integrator.clone();
  cloned->integrate(1.0);
  integrator.integrate(1.0);

  std::vector<std::complex<double>> origVec(4), cloneVec(4);
  auto [t1, origState] = integrator.getState();
  origState.to_host(origVec.data(), origVec.size());
  auto [t2, cloneState] = cloned->getState();
  cloneState.to_host(cloneVec.data(), cloneVec.size());

  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(origVec[i].real(), cloneVec[i].real(), 1e-10);
    EXPECT_NEAR(origVec[i].imag(), cloneVec[i].imag(), 1e-10);
  }
}

// Open-system relaxation (T1 decay): with a collapse operator present, the CF4
// integrator transparently falls back to magnus_expansion. Verify <n> decays as
// exp(-gamma t) through the high-level evolve path.
TEST(MagnusHighOrderIntegratorEvolve, T1DecayFallback) {
  const cudaq::dimension_map dims = {{0, 2}};
  cudaq::product_op<cudaq::matrix_handler> ham_t =
      0.0 * cudaq::sum_op<cudaq::matrix_handler>::number(0);
  cudaq::sum_op<cudaq::matrix_handler> hamiltonian(ham_t);

  constexpr int numSteps = 11;
  std::vector<double> timeSteps = cudaq::linspace(0.0, 5.0, numSteps);
  cudaq::schedule schedule(timeSteps, {"t"});

  Eigen::MatrixXcd rho0 = Eigen::MatrixXcd::Zero(2, 2);
  rho0(1, 1) = 1.0;
  auto initialState =
      cudaq::state::from_data(std::make_pair(rho0.data(), rho0.size()));

  cudaq::integrators::magnus_cf4 integrator(0.01);

  constexpr double decayRate = 0.1;
  cudaq::product_op<cudaq::matrix_handler> collapseOp_t =
      std::sqrt(decayRate) * cudaq::boson_op::annihilate(0);
  cudaq::sum_op<cudaq::matrix_handler> collapseOp(collapseOp_t);

  cudaq::product_op<cudaq::matrix_handler> occ_t =
      cudaq::sum_op<cudaq::matrix_handler>::number(0);
  cudaq::sum_op<cudaq::matrix_handler> occ(occ_t);

  cudaq::evolve_result result = cudaq::detail::evolveSingle(
      hamiltonian, dims, schedule, initialState, integrator, {collapseOp},
      {occ}, cudaq::IntermediateResultSave::All);
  ASSERT_TRUE(result.expectation_values.has_value());
  ASSERT_EQ(result.expectation_values.value().size(), numSteps);

  int idx = 0;
  for (auto expVals : result.expectation_values.value()) {
    const double time = timeSteps[idx++];
    const double expected = std::exp(-decayRate * time);
    EXPECT_NEAR((double)expVals[0], expected, 0.05)
        << "T1 decay mismatch at t=" << time;
  }
}
