// /*******************************************************************************
//  * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "CuDensityMatState.h"
#include "CuDensityMatTimeStepper.h"
#include "cudaq/dynamics_integrators.h"
#include "test_Mocks.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

using namespace cudaq;

class RungeKuttaIntegratorTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle_;
  cudensitymatOperator_t liouvillian_;
  std::unique_ptr<RungeKuttaIntegrator> integrator_;
  std::unique_ptr<CuDensityMatState> state_;

  void SetUp() override {
    // Create library handle
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_));

    // Create a mock Liouvillian
    liouvillian_ = mock_liouvillian(handle_);

    // Create initial state
    state_ = std::make_unique<CuDensityMatState>(
        handle_, mock_initial_state_data(), mock_hilbert_space_dims());
    ASSERT_NE(state_, nullptr);
    ASSERT_TRUE(state_->is_initialized());

    double t0 = 0.0;
    // Initialize the integrator (using substeps = 4, for Runge-Kutta method)
    ASSERT_NO_THROW(integrator_ = std::make_unique<RungeKuttaIntegrator>());
    ASSERT_NE(integrator_, nullptr);
    integrator_->order = 4;
  }

  void TearDown() override {
    // Clean up resources
    HANDLE_CUDM_ERROR(cudensitymatDestroyOperator(liouvillian_));
    HANDLE_CUDM_ERROR(cudensitymatDestroy(handle_));
  }
};

// Test Initialization
TEST_F(RungeKuttaIntegratorTest, Initialization) {
  ASSERT_NE(integrator_, nullptr);
}

TEST_F(RungeKuttaIntegratorTest, CheckEvolve) {
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  auto spin_op_x = cudaq::spin_operator::x(0);
  cudaq::product_operator<cudaq::matrix_operator> ham1 =
      2.0 * M_PI * 0.1 * spin_op_x;
  cudaq::operator_sum<cudaq::matrix_operator> ham(ham1);
  SystemDynamics system;
  system.hamiltonian = &ham;
  system.modeExtents = dims;

  for (int integratorOrder : {1, 2, 4}) {
    std::cout << "Test RK order " << integratorOrder << "\n";
    auto integrator = std::make_shared<cudaq::RungeKuttaIntegrator>();
    integrator->dt = 0.001;
    integrator->order = integratorOrder;
    constexpr std::size_t numDataPoints = 10;
    double t = 0.0;
    auto initialState = cudaq::state::from_data(initialStateVec);
    // initialState.dump();
    auto *simState = cudaq::state_helper::getSimulationState(&initialState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    EXPECT_TRUE(castSimState != nullptr);
    castSimState->initialize_cudm(handle_, dims);
    integrator->setState(initialState, 0.0);
    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0.0, 1.0 * numDataPoints, numDataPoints)) {
      steps.emplace_back(t, 0.0);
    }
    cudaq::Schedule schedule(
        steps, {"t"}, [](const std::string &, const std::complex<double> &val) {
          return val;
        });
    integrator->setSystem(system, schedule);
    std::vector<std::complex<double>> outputStateVec(2);
    for (std::size_t i = 1; i < numDataPoints; ++i) {
      integrator->integrate(i);
      auto [t, state] = integrator->getState();
      // std::cout << "Time = " << t << "\n";
      // state.dump();
      state.to_host(outputStateVec.data(), outputStateVec.size());
      // Check state vector norm
      EXPECT_NEAR(std::norm(outputStateVec[0]) + std::norm(outputStateVec[1]),
                  1.0, 1e-2);
      const double expValZ =
          std::norm(outputStateVec[0]) - std::norm(outputStateVec[1]);
      // Analytical results
      EXPECT_NEAR(outputStateVec[0].real(), std::cos(2.0 * M_PI * 0.1 * t),
                  1e-2);
    }
  }

  // Add test to test tensor_callback
}
