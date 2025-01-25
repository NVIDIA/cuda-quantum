// /*******************************************************************************
//  * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "cudaq/runge_kutta_integrator.h"
#include "runge_kutta_test_helpers.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

using namespace cudaq;

// Test fixture class
class RungeKuttaIntegratorTest : public ::testing::Test {
protected:
  RungeKuttaIntegrator<TestState> *integrator;
  std::shared_ptr<Schedule> schedule;
  std::shared_ptr<operator_sum> hamiltonian;

  void SetUp() override {
    integrator = new RungeKuttaIntegrator<TestState>(simple_derivative);
    // Initial state and time
    integrator->set_state(1.0, 0.0);

    // A simple step sequence for the schedule
    std::vector<std::complex<double>> steps = {0.1, 0.2, 0.3, 0.4, 0.5};

    // Dummy parameters
    std::vector<std::string> parameters = {"param1"};

    // A simple parameter function
    auto value_function = [](const std::string &param,
                             const std::complex<double> &step) { return step; };

    // A valid schedule instance
    schedule = std::make_shared<Schedule>(steps, parameters, value_function);

    // A simple hamiltonian as an operator_sum
    hamiltonian = std::make_shared<operator_sum>();
    *hamiltonian += 0.5 * elementary_operator::identity(0);
    *hamiltonian += 0.5 * elementary_operator::number(0);

    // System with valid components
    integrator->set_system({{0, 2}}, schedule, hamiltonian);
  }

  void TearDown() override { delete integrator; }
};

// Basic integration
TEST_F(RungeKuttaIntegratorTest, BasicIntegration) {
  integrator->integrate(1.0);

  // Expected result: x(t) = e^(-t)
  double expected = std::exp(-1.0);

  EXPECT_NEAR(integrator->get_state().second, expected, 1e-3)
      << "Basic Runge-Kutta integration failed!";
}

// Time evolution
TEST_F(RungeKuttaIntegratorTest, TimeEvolution) {
  integrator->integrate(2.0);

  double expected = 2.0;

  EXPECT_NEAR(integrator->get_state().first, expected, 1e-5)
      << "Integrator did not correctly update time!";
}

// Large step size
TEST_F(RungeKuttaIntegratorTest, LargeStepSize) {
  integrator->integrate(5.0);

  double expected = std::exp(-5.0);

  EXPECT_NEAR(integrator->get_state().second, expected, 1e-1)
      << "Runge-Kutta integration failed for large step size!!";
}

// // Integrating Sine function
// TEST_F(RungeKuttaIntegratorTest, SineFunction) {
//     integrator = new RungeKuttaIntegrator<TestState>(sine_derivative);
//     integrator->set_state(1.0, 0.0);
//     integrator->set_system({{0, 2}}, schedule, hamiltonian);

//     integrator->integrate(M_PI / 2);

//     double expected = std::cos(M_PI / 2);

//     EXPECT_NEAR(integrator->get_state().second, expected, 1e-3) <<
//     "Runge-Kutta integration for sine function failed!";
// }

// Small step size
TEST_F(RungeKuttaIntegratorTest, SmallStepIntegration) {
  integrator->set_state(1.0, 0.0);
  integrator->set_system({{0, 2}}, schedule, hamiltonian);

  double step_size = 0.001;
  while (integrator->get_state().first < 1.0) {
    integrator->integrate(integrator->get_state().first + step_size);
  }

  double expected = std::exp(-1.0);

  EXPECT_NEAR(integrator->get_state().second, expected, 5e-4)
      << "Runge-Kutta integration for small step size failed!";
}

// Large step size
TEST_F(RungeKuttaIntegratorTest, LargeStepIntegration) {
  integrator->set_state(1.0, 0.0);
  integrator->set_system({{0, 2}}, schedule, hamiltonian);

  double step_size = 0.5;
  double t = 0.0;
  double target_t = 1.0;
  while (t < target_t) {
    integrator->integrate(std::min(t + step_size, target_t));
    t += step_size;
  }

  double expected = std::exp(-1.0);

  EXPECT_NEAR(integrator->get_state().second, expected, 1e-2)
      << "Runge-Kutta integration for large step size failed!";
}
