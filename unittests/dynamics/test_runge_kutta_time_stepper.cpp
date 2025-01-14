/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "runge_kutta_test_helpers.h"
#include "cudaq/runge_kutta_time_stepper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <cmath>

// Test fixture class
class RungeKuttaTimeStepperTest : public ::testing::Test {
protected:
    std::shared_ptr<cudaq::RungeKuttaTimeStepper<TestState>> stepper;

    void SetUp() override {
        stepper = std::make_shared<cudaq::RungeKuttaTimeStepper<TestState>>(simple_derivative);
    }
};

// Single step integration
TEST_F(RungeKuttaTimeStepperTest, SingleStep) {
    // Initial values
    double state = 1.0;
    double t = 0.0;
    double dt = 0.1;

    stepper->compute(state, t, dt);

    // Expected result using analytical solution: x(t) = e^(-t)
    double expected = std::exp(-dt);

    EXPECT_NEAR(state, expected, 1e-3) << "Single step Runge-Kutta integration failed!";
}

// Multiple step integration
TEST_F(RungeKuttaTimeStepperTest, MultipleSteps) {
    // Initial values
    double state = 1.0;
    double t = 0.0;
    double dt = 0.1;
    int steps = 10;

    for (int i = 0; i < steps; i++) {
        stepper->compute(state, t, dt);
    }

    // Expected result: x(t) = e^(-t)
    double expected = std::exp(-1.0);

    EXPECT_NEAR(state, expected, 1e-2) << "Multiple step Runge-Kutta integration failed!";
}

// Convergence to Analytical Solution
TEST_F(RungeKuttaTimeStepperTest, Convergence) {
    // Initial values
    double state = 1.0;
    double t = 0.0;
    double dt = 0.01;
    int steps = 100;

    for (int i = 0; i < steps; i++) {
        stepper->compute(state, t, dt);
    }

    double expected = std::exp(-1.0);

    EXPECT_NEAR(state, expected, 1e-3) << "Runge-Kutta integration does not converge!";
}

// // Integrating Sine function
// TEST_F(RungeKuttaTimeStepperTest, SineFunction) {
//     auto sine_stepper = std::make_shared<cudaq::RungeKuttaTimeStepper<TestState>>(sine_derivative);

//     // Initial values
//     double state = 0.0;
//     double t = 0.0;
//     double dt = 0.1;
//     int steps = 10;

//     for (int i = 0; i < steps; i++) {
//         sine_stepper->compute(state, t, dt);
//     }

//     // Expected integral of sin(t) over [0, 1] is 1 - cos(1)
//     double expected = 1 - std::cos(1);

//     EXPECT_NEAR(state, expected, 1e-2) << "Runge-Kutta integration for sine function failed!";
// }

// Handling small steps sizes
TEST_F(RungeKuttaTimeStepperTest, SmallStepSize) {
    // Initial values
    double state = 1.0;
    double t = 0.0;
    double dt = 1e-5;
    int steps = 100000;

    for (int i = 0; i < steps; i++) {
        stepper->compute(state, t, dt);
    }

    double expected = std::exp(-1.0);

    EXPECT_NEAR(state, expected, 1e-3) << "Runge-Kutta fails with small step sizes!";
}

// Handling large steps sizes
TEST_F(RungeKuttaTimeStepperTest, LargeStepSize) {
    // Initial values
    double state = 1.0;
    double t = 0.0;
    double dt = 1.0;

    stepper->compute(state, t, dt);

    double expected = std::exp(-1.0);

    EXPECT_NEAR(state, expected, 1e-1) << "Runge-Kutta is unstable with large step sizes!";
}

// Constant derivative (dx/dt = 0)
TEST_F(RungeKuttaTimeStepperTest, ConstantFunction) {
    auto constant_stepper = std::make_shared<cudaq::RungeKuttaTimeStepper<TestState>>(
        [](const TestState &state, double t) {
            return 0.0;
        }
    );

    // Initial values
    double state = 5.0;
    double t = 0.0;
    double dt = 0.1;
    int steps = 10;

    for (int i = 0; i < steps; i++) {
        constant_stepper->compute(state, t, dt);
    }

    EXPECT_NEAR(state, 5.0, 1e-6) << "Runge-Kutta should not change a constant function!";
}



