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
#include "cudaq/algorithms/integrator.h"
#include "test_Mocks.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

using namespace cudaq;

class MagnusIntegratorTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle_;
  cudensitymatOperator_t liouvillian_;

  void SetUp() override {
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_));
    liouvillian_ = mock_liouvillian(handle_);
  }

  void TearDown() override {
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(liouvillian_));
    HANDLE_CUDM_ERROR(cudensitymatDestroy(handle_));
  }
};

TEST_F(MagnusIntegratorTest, Initialization) {
  EXPECT_NO_THROW(cudaq::integrators::magnus_expansion m1);
  EXPECT_NO_THROW(cudaq::integrators::magnus_expansion m2(5));
  EXPECT_NO_THROW(cudaq::integrators::magnus_expansion m3(15, 0.01));
  EXPECT_THROW(cudaq::integrators::magnus_expansion bad(0),
               std::invalid_argument);
}

TEST_F(MagnusIntegratorTest, CheckEvolve) {
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  cudaq::sum_op<cudaq::matrix_handler> ham(2.0 * M_PI * 0.1 *
                                           cudaq::spin_op::x(0));
  SystemDynamics system(dims, ham);

  cudaq::integrators::magnus_expansion integrator(10, 0.001);

  auto initialState = cudaq::state::from_data(initialStateVec);
  auto *castSimState = dynamic_cast<CuDensityMatState *>(
      cudaq::state_helper::getSimulationState(&initialState));
  ASSERT_NE(castSimState, nullptr);
  castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);

  integrator.setState(initialState, 0.0);

  constexpr std::size_t numDataPoints = 10;
  std::vector<std::complex<double>> steps;
  for (double t : cudaq::linspace(0.0, 1.0 * numDataPoints, numDataPoints))
    steps.emplace_back(t, 0.0);

  cudaq::schedule schedule(
      steps, {"t"},
      [](const std::string &, const std::complex<double> &val) { return val; });
  cudaq::integrator_helper::init_system_dynamics(integrator, system, schedule);

  std::vector<std::complex<double>> outVec(2);
  for (std::size_t i = 1; i < numDataPoints; ++i) {
    integrator.integrate(static_cast<double>(i));
    auto [t, state] = integrator.getState();
    state.to_host(outVec.data(), outVec.size());

    EXPECT_NEAR(std::norm(outVec[0]) + std::norm(outVec[1]), 1.0, 1e-2);
    EXPECT_NEAR(outVec[0].real(), std::cos(2.0 * M_PI * 0.1 * t), 1e-2);
  }
}

TEST_F(MagnusIntegratorTest, CloneReproducesTrajectory) {
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  cudaq::sum_op<cudaq::matrix_handler> ham(2.0 * M_PI * 0.1 *
                                           cudaq::spin_op::x(0));
  SystemDynamics system(dims, ham);

  cudaq::integrators::magnus_expansion integrator(10, 0.01);
  auto initialState = cudaq::state::from_data(initialStateVec);
  auto *castSimState = dynamic_cast<CuDensityMatState *>(
      cudaq::state_helper::getSimulationState(&initialState));
  ASSERT_NE(castSimState, nullptr);
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

TEST_F(MagnusIntegratorTest, ConvergenceOrderVerification) {
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  const double omega = 2.0 * M_PI * 0.1;
  cudaq::sum_op<cudaq::matrix_handler> ham(omega * cudaq::spin_op::x(0));
  SystemDynamics system(dims, ham);

  const double t_final = 5.0;
  constexpr std::size_t numDataPoints = 51;

  auto runEvolution = [&](double stepSize) -> double {
    cudaq::integrators::magnus_expansion integrator(3, stepSize);
    auto initialState = cudaq::state::from_data(initialStateVec);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(
        cudaq::state_helper::getSimulationState(&initialState));
    castSimState->initialize_cudm(handle_, dims, 1);
    integrator.setState(initialState, 0.0);

    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0.0, t_final, numDataPoints))
      steps.emplace_back(t, 0.0);
    cudaq::schedule schedule(
        steps, {"t"}, [](const std::string &, const std::complex<double> &val) {
          return val;
        });
    cudaq::integrator_helper::init_system_dynamics(integrator, system,
                                                   schedule);
    integrator.integrate(t_final);
    auto [t, state] = integrator.getState();
    std::vector<std::complex<double>> outVec(2);
    state.to_host(outVec.data(), outVec.size());
    double analytical = std::cos(omega * t_final) * std::cos(omega * t_final);
    return std::abs(std::norm(outVec[0]) - analytical);
  };

  const double h1 = 0.1;
  const double h2 = 0.05;
  double err1 = runEvolution(h1);
  double err2 = runEvolution(h2);
  double ratio = err1 / err2;
  double estimated_order = std::log2(ratio);

  std::cout << "Magnus convergence (3 Taylor terms): err(h=" << h1
            << ")=" << err1 << ", err(h=" << h2 << ")=" << err2
            << ", ratio=" << ratio << ", estimated order=" << estimated_order
            << "\n";

  // Expect ≥ 2nd order: ratio ≈ 4; accept at least half of that.
  EXPECT_GE(ratio, 2.0)
      << "Magnus expansion should show at least 2nd-order convergence";
  EXPECT_GE(estimated_order, 1.5)
      << "Estimated order should be close to 2 for 2nd-order Magnus";
}

TEST_F(MagnusIntegratorTest, MoreTaylorTermsImproveAccuracy) {
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  const double omega = 2.0 * M_PI * 0.5;
  cudaq::sum_op<cudaq::matrix_handler> ham(omega * cudaq::spin_op::x(0));
  SystemDynamics system(dims, ham);

  const double t_final = 1.0;
  constexpr std::size_t numDataPoints = 11;
  const double step_size = 0.2;

  auto computeError = [&](int numTerms) -> double {
    cudaq::integrators::magnus_expansion integrator(numTerms, step_size);
    auto initialState = cudaq::state::from_data(initialStateVec);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(
        cudaq::state_helper::getSimulationState(&initialState));
    castSimState->initialize_cudm(handle_, dims, 1);
    integrator.setState(initialState, 0.0);

    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0.0, t_final, numDataPoints))
      steps.emplace_back(t, 0.0);
    cudaq::schedule schedule(
        steps, {"t"}, [](const std::string &, const std::complex<double> &val) {
          return val;
        });
    cudaq::integrator_helper::init_system_dynamics(integrator, system,
                                                   schedule);
    integrator.integrate(t_final);
    auto [t, state] = integrator.getState();
    std::vector<std::complex<double>> outVec(2);
    state.to_host(outVec.data(), outVec.size());
    double analytical = std::cos(omega * t_final) * std::cos(omega * t_final);
    return std::abs(std::norm(outVec[0]) - analytical);
  };

  double err_low = computeError(3);
  double err_high = computeError(15);

  std::cout << "Magnus Taylor terms comparison (step=" << step_size
            << ", omega=" << omega << "):\n"
            << "  3 terms error:  " << err_low << "\n"
            << "  15 terms error: " << err_high << "\n";

  EXPECT_LT(err_high, err_low)
      << "More Taylor terms should yield smaller error";
}

TEST_F(MagnusIntegratorTest, NormPreservationLongEvolution) {
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  cudaq::sum_op<cudaq::matrix_handler> ham(2.0 * M_PI * 0.1 *
                                           cudaq::spin_op::x(0));
  SystemDynamics system(dims, ham);

  cudaq::integrators::magnus_expansion integrator(10, 0.01);

  auto initialState = cudaq::state::from_data(initialStateVec);
  auto *castSimState = dynamic_cast<CuDensityMatState *>(
      cudaq::state_helper::getSimulationState(&initialState));
  ASSERT_NE(castSimState, nullptr);
  castSimState->initialize_cudm(handle_, dims, 1);

  std::vector<std::complex<double>> steps;
  for (double t : cudaq::linspace(0.0, 10.0, 101))
    steps.emplace_back(t, 0.0);
  cudaq::schedule schedule(
      steps, {"t"},
      [](const std::string &, const std::complex<double> &v) { return v; });

  integrator.setState(initialState, 0.0);
  cudaq::integrator_helper::init_system_dynamics(integrator, system, schedule);

  std::vector<std::complex<double>> outVec(2);
  for (int i = 1; i <= 10; ++i) {
    integrator.integrate(static_cast<double>(i));
    auto [t, state] = integrator.getState();
    state.to_host(outVec.data(), outVec.size());
    double norm = std::norm(outVec[0]) + std::norm(outVec[1]);
    EXPECT_NEAR(norm, 1.0, 1e-2) << "Norm should be preserved at t=" << t;
  }
}
