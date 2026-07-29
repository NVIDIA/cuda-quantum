// /*******************************************************************************
//  * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "CuDensityMatState.h"
#include "CuDensityMatTimeStepper.h"
#include "CuDensityMatUtils.h"
#include "common/EigenDense.h"
#include "cudaq/algorithms/evolve_internal.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/utils/cudaq_utils.h"
#include "test_Mocks.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

using namespace cudaq;

namespace {
// Drive a single qubit with H = 2*pi*0.1*X and return the evolved amplitudes.
// Sets up the cuDensityMat state and schedule the same way as the other
// integrator tests, and returns the final dopri5 statistics for inspection.
struct RabiResult {
  std::vector<std::complex<double>> amps;
  cudaq::integrators::dopri5::Stats stats;
};
} // namespace

class AdaptiveIntegratorTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle_;

  void SetUp() override { HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_)); }
  void TearDown() override { HANDLE_CUDM_ERROR(cudensitymatDestroy(handle_)); }

  RabiResult runRabi(cudaq::integrators::dopri5 &integrator, double tFinal,
                     std::size_t numDataPoints) {
    const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                               {0.0, 0.0}};
    const std::vector<int64_t> dims = {2};
    cudaq::sum_op<cudaq::matrix_handler> ham(2.0 * M_PI * 0.1 *
                                             cudaq::spin_op::x(0));
    SystemDynamics system(dims, ham);

    auto initialState = cudaq::state::from_data(initialStateVec);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(
        cudaq::state_helper::getSimulationState(&initialState));
    EXPECT_NE(castSimState, nullptr);
    castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);

    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0.0, tFinal, numDataPoints))
      steps.emplace_back(t, 0.0);
    cudaq::schedule schedule(
        steps, {"t"},
        [](const std::string &, const std::complex<double> &v) { return v; });

    integrator.setState(initialState, 0.0);
    cudaq::integrator_helper::init_system_dynamics(integrator, system,
                                                   schedule);
    integrator.integrate(tFinal);
    auto [t, state] = integrator.getState();
    std::vector<std::complex<double>> outVec(2);
    state.to_host(outVec.data(), outVec.size());
    return {outVec, integrator.getStats()};
  }
};

TEST_F(AdaptiveIntegratorTest, Initialization) {
  EXPECT_NO_THROW(cudaq::integrators::dopri5 d1);
  EXPECT_NO_THROW(cudaq::integrators::dopri5 d2(1e-8, 1e-10));
  EXPECT_NO_THROW(cudaq::integrators::dopri5 d3(1e-6, 1e-8, 0.05, 1e-7, 0.5));
  EXPECT_THROW(cudaq::integrators::dopri5 bad(-1.0, 1e-8),
               std::invalid_argument);
  EXPECT_THROW(cudaq::integrators::dopri5 bad(1e-6, 0.0), std::invalid_argument);
  EXPECT_THROW(cudaq::integrators::dopri5 bad(1e-6, 1e-8, 0.01, 1.0, 0.1),
               std::invalid_argument);
}

TEST_F(AdaptiveIntegratorTest, CheckEvolve) {
  cudaq::integrators::dopri5 integrator(1e-8, 1e-10, 0.01, 1e-6, 1.0);
  auto result = runRabi(integrator, 10.0, 11);
  const auto &outVec = result.amps;
  const double t = 10.0;
  EXPECT_NEAR(std::norm(outVec[0]) + std::norm(outVec[1]), 1.0, 1e-4);
  EXPECT_NEAR(outVec[0].real(), std::cos(2.0 * M_PI * 0.1 * t), 1e-4);
}

TEST_F(AdaptiveIntegratorTest, AdaptiveStatsSanity) {
  cudaq::integrators::dopri5 integrator(1e-7, 1e-9, 0.01, 1e-6, 1.0);
  auto result = runRabi(integrator, 10.0, 11);
  const auto &stats = result.stats;

  EXPECT_GT(stats.accepted_steps, 0u);
  EXPECT_LE(stats.min_dt_used, stats.avg_dt + 1e-12);
  EXPECT_LE(stats.avg_dt, stats.max_dt_used + 1e-12);

  const double totalSteps =
      static_cast<double>(stats.accepted_steps + stats.rejected_steps);
  const double rejectionRate =
      totalSteps > 0 ? stats.rejected_steps / totalSteps : 0.0;
  std::cout << "dopri5 stats: accepted=" << stats.accepted_steps
            << " rejected=" << stats.rejected_steps
            << " rejectionRate=" << rejectionRate
            << " min_dt=" << stats.min_dt_used << " avg_dt=" << stats.avg_dt
            << " max_dt=" << stats.max_dt_used << "\n";
  EXPECT_LT(rejectionRate, 0.2)
      << "Adaptive step rejection rate should be modest for smooth dynamics";
}

TEST_F(AdaptiveIntegratorTest, MatchesRungeKutta4) {
  // Adaptive dopri5 at tight tolerance should agree with a well-resolved RK4.
  cudaq::integrators::dopri5 adaptive(1e-9, 1e-11, 0.01, 1e-7, 0.5);
  auto adaptiveResult = runRabi(adaptive, 5.0, 6);

  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  cudaq::sum_op<cudaq::matrix_handler> ham(2.0 * M_PI * 0.1 *
                                           cudaq::spin_op::x(0));
  SystemDynamics system(dims, ham);

  cudaq::integrators::runge_kutta rk(4, 0.001);
  auto initialState = cudaq::state::from_data(initialStateVec);
  auto *castSimState = dynamic_cast<CuDensityMatState *>(
      cudaq::state_helper::getSimulationState(&initialState));
  castSimState->initialize_cudm(handle_, dims, 1);
  std::vector<std::complex<double>> steps;
  for (double t : cudaq::linspace(0.0, 5.0, 6))
    steps.emplace_back(t, 0.0);
  cudaq::schedule schedule(
      steps, {"t"},
      [](const std::string &, const std::complex<double> &v) { return v; });
  rk.setState(initialState, 0.0);
  cudaq::integrator_helper::init_system_dynamics(rk, system, schedule);
  rk.integrate(5.0);
  auto [t, rkState] = rk.getState();
  std::vector<std::complex<double>> rkVec(2);
  rkState.to_host(rkVec.data(), rkVec.size());

  EXPECT_NEAR(adaptiveResult.amps[0].real(), rkVec[0].real(), 1e-6);
  EXPECT_NEAR(adaptiveResult.amps[0].imag(), rkVec[0].imag(), 1e-6);
  EXPECT_NEAR(adaptiveResult.amps[1].real(), rkVec[1].real(), 1e-6);
  EXPECT_NEAR(adaptiveResult.amps[1].imag(), rkVec[1].imag(), 1e-6);
}

TEST_F(AdaptiveIntegratorTest, CloneReproducesTrajectory) {
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  cudaq::sum_op<cudaq::matrix_handler> ham(2.0 * M_PI * 0.1 *
                                           cudaq::spin_op::x(0));
  SystemDynamics system(dims, ham);

  cudaq::integrators::dopri5 integrator(1e-8, 1e-10, 0.01, 1e-6, 1.0);
  auto initialState = cudaq::state::from_data(initialStateVec);
  auto *castSimState = dynamic_cast<CuDensityMatState *>(
      cudaq::state_helper::getSimulationState(&initialState));
  castSimState->initialize_cudm(handle_, dims, 1);
  std::vector<std::complex<double>> steps;
  for (double t : cudaq::linspace(0.0, 1.0, 11))
    steps.emplace_back(t, 0.0);
  cudaq::schedule schedule(
      steps, {"t"},
      [](const std::string &, const std::complex<double> &v) { return v; });

  integrator.setState(initialState, 0.0);
  cudaq::integrator_helper::init_system_dynamics(integrator, system, schedule);
  integrator.integrate(0.5);

  auto cloned = integrator.clone();
  cloned->integrate(1.0);
  integrator.integrate(1.0);

  std::vector<std::complex<double>> origVec(2), cloneVec(2);
  auto [t1, origState] = integrator.getState();
  origState.to_host(origVec.data(), origVec.size());
  auto [t2, cloneState] = cloned->getState();
  cloneState.to_host(cloneVec.data(), cloneVec.size());

  EXPECT_NEAR(origVec[0].real(), cloneVec[0].real(), 1e-10);
  EXPECT_NEAR(origVec[0].imag(), cloneVec[0].imag(), 1e-10);
  EXPECT_NEAR(origVec[1].real(), cloneVec[1].real(), 1e-10);
  EXPECT_NEAR(origVec[1].imag(), cloneVec[1].imag(), 1e-10);
}

// Open-system relaxation (T1 decay) through the high-level evolve path,
// verifying <n> decays as exp(-gamma t).
TEST(AdaptiveIntegratorEvolve, T1Decay) {
  const cudaq::dimension_map dims = {{0, 2}};
  // No coherent drive: pure relaxation.
  cudaq::product_op<cudaq::matrix_handler> ham_t =
      0.0 * cudaq::matrix_handler::number(0);
  cudaq::sum_op<cudaq::matrix_handler> hamiltonian(ham_t);

  constexpr int numSteps = 11;
  std::vector<double> timeSteps = cudaq::linspace(0.0, 5.0, numSteps);
  cudaq::schedule schedule(timeSteps, {"t"});

  // Start in the excited state |1><1| as a density matrix.
  Eigen::MatrixXcd rho0 = Eigen::MatrixXcd::Zero(2, 2);
  rho0(1, 1) = 1.0;
  auto initialState =
      cudaq::state::from_data(std::make_pair(rho0.data(), rho0.size()));

  cudaq::integrators::dopri5 integrator(1e-8, 1e-10, 0.01, 1e-6, 0.5);

  constexpr double decayRate = 0.1;
  cudaq::product_op<cudaq::matrix_handler> collapseOp_t =
      std::sqrt(decayRate) * cudaq::boson_op::annihilate(0);
  cudaq::sum_op<cudaq::matrix_handler> collapseOp(collapseOp_t);

  cudaq::product_op<cudaq::matrix_handler> occ_t =
      cudaq::matrix_handler::number(0);
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
