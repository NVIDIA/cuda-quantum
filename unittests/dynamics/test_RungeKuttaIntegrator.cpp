// /*******************************************************************************
//  * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "CuDensityMatIntegratorBase.h"
#include "CuDensityMatState.h"
#include "CuDensityMatTimeStepper.h"
#include "CuDensityMatUtils.h"
#include "test_Mocks.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/utils/cudaq_utils.h"
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <optional>

using namespace cudaq;

class RungeKuttaIntegratorTest : public ::testing::Test {
protected:
  cudensitymatHandle_t handle_;
  cudensitymatOperator_t liouvillian_;
  std::unique_ptr<cudaq::integrators::runge_kutta> integrator_;
  std::unique_ptr<CuDensityMatState> state_;

  void SetUp() override {
    // Create library handle
    HANDLE_CUDM_ERROR(cudensitymatCreate(&handle_));

    // Create a mock Liouvillian
    liouvillian_ = mock_liouvillian(handle_);

    // Create initial state
    state_ = std::make_unique<CuDensityMatState>(
        mock_initial_state_data().size(),
        cudaq::dynamics::createArrayGpu(mock_initial_state_data()));
    state_->initialize_cudm(handle_, mock_hilbert_space_dims(),
                            /*batchSize=*/1);
    ASSERT_NE(state_, nullptr);
    ASSERT_TRUE(state_->is_initialized());

    double t0 = 0.0;
    // Initialize the integrator (using substeps = 4, for Runge-Kutta method)
    ASSERT_NO_THROW(integrator_ =
                        std::make_unique<cudaq::integrators::runge_kutta>());
    ASSERT_NE(integrator_, nullptr);
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
  EXPECT_THROW(cudaq::integrators::runge_kutta(4, 0.0), std::invalid_argument);
  EXPECT_THROW(cudaq::integrators::runge_kutta(4, -0.01),
               std::invalid_argument);
}

namespace {
/// Walk one schedule the way the integrators do and report what happened.
struct SubStepTrace {
  std::size_t count = 0;
  double endTime = 0.0;
  double maxStep = 0.0;
  double minStep = std::numeric_limits<double>::max();
  bool landedOnEveryPoint = true;
};

/// Sub-step boundaries are quantised to the timestamps, so consecutive
/// sub-steps cannot be exactly equal; a single one may run over by an ulp.
double largestAllowedSubStep(const std::vector<double> &points,
                             double maxStepSize) {
  const double scale =
      std::max(std::abs(points.front()), std::abs(points.back()));
  return maxStepSize +
         (std::nextafter(scale, std::numeric_limits<double>::max()) - scale);
}

SubStepTrace walkSchedule(const std::vector<double> &points,
                          const std::optional<double> &maxStepSize) {
  SubStepTrace trace;
  double currentTime = points.front();
  for (std::size_t i = 1; i < points.size(); ++i) {
    const double targetTime = points[i];
    const double startTime = currentTime;
    const auto numSubSteps = CuDensityMatIntegratorHelper::subStepCount(
        startTime, targetTime, maxStepSize);
    for (std::int64_t subStep = 1; subStep <= numSubSteps; ++subStep) {
      const double nextTime = CuDensityMatIntegratorHelper::subStepTime(
          startTime, targetTime, subStep, numSubSteps);
      const double stepSize = nextTime - currentTime;
      trace.maxStep = std::max(trace.maxStep, stepSize);
      trace.minStep = std::min(trace.minStep, stepSize);
      currentTime = nextTime;
      ++trace.count;
    }
    if (currentTime != targetTime)
      trace.landedOnEveryPoint = false;
  }
  trace.endTime = currentTime;
  return trace;
}
} // namespace

// A max step size that nominally divides the schedule spacing must not produce
// a redundant near-zero sub-step.
//
// The integrators used to walk an interval by accumulating time:
//
//   while (m_t < targetTime) {
//     step = std::min(maxStepSize, targetTime - m_t);
//     ...propagate...
//     m_t += step;
//   }
//
// `std::min` only selects the clamp when it is strictly below the max step, so
// when the two are within a rounding error of each other it returns the max
// step instead and `m_t` lands an ulp short of the target. The loop condition
// still holds, so the next iteration spends a full right-hand-side evaluation
// on a ~1e-17 step. On the schedule below, 22 of the 99 points did this,
// costing 121 sub-steps of which 22 were pure waste.
//
// Separately, most of these intervals measure under an ulp wider than one whole
// step, which a bare ceil() would cover with two sub-steps each. The rounding
// allowance in subStepCount() keeps them at one.
TEST(CuDensityMatIntegratorHelperTest, NominallyEqualStepsReachTargetExactly) {
  constexpr std::size_t numIntervals = 99;
  const double maxStepSize = 1.0 / numIntervals;
  std::vector<double> points;
  for (std::size_t i = 0; i <= numIntervals; ++i)
    points.push_back(static_cast<double>(i) / numIntervals);

  const auto trace = walkSchedule(points, maxStepSize);

  EXPECT_TRUE(trace.landedOnEveryPoint);
  EXPECT_EQ(trace.endTime, 1.0);
  EXPECT_LE(trace.maxStep, largestAllowedSubStep(points, maxStepSize));
  EXPECT_EQ(trace.count, numIntervals);
  EXPECT_GT(trace.minStep, 0.4 * maxStepSize);
}

// Sub-step boundaries are interpolated, never accumulated, so the schedule is
// hit exactly regardless of the absolute time origin or the time scale.
TEST(CuDensityMatIntegratorHelperTest, LandsExactlyAcrossTimeScalesAndOrigins) {
  const std::vector<std::pair<double, double>> spans = {
      {0.0, 1.0},  {1.0, 2.0},   {100.0, 101.0},
      {-1.0, 1.0}, {0.0, 1e-14}, {1e6, 1e6 + 1.0}};
  for (const auto &[from, to] : spans) {
    for (const std::size_t numIntervals : {7u, 13u, 100u}) {
      for (const std::size_t subStepsPerInterval : {1u, 3u, 10u}) {
        std::vector<double> points;
        for (std::size_t i = 0; i <= numIntervals; ++i)
          points.push_back(from + (to - from) *
                                      (static_cast<double>(i) / numIntervals));
        const double maxStepSize =
            (points[1] - points[0]) / subStepsPerInterval;
        const auto trace = walkSchedule(points, maxStepSize);

        EXPECT_TRUE(trace.landedOnEveryPoint)
            << "span [" << from << ", " << to << "] intervals " << numIntervals;
        EXPECT_EQ(trace.endTime, points.back());
        EXPECT_LE(trace.maxStep, largestAllowedSubStep(points, maxStepSize));
        // No degenerate sub-step: the loop never spends an evaluation on a
        // step that is orders of magnitude below the requested one.
        EXPECT_GT(trace.minStep, 0.4 * maxStepSize);
      }
    }
  }
}

// A remaining interval just above the max step size must be split, not
// absorbed, which would hand the propagator a step wider than the caller asked
// for. Sizes here are in ulps of the start time, so the max step size is only a
// handful of ulps -- the regime where the rounding allowance is largest
// relative to the step, and so the easiest place to get this wrong.
TEST(CuDensityMatIntegratorHelperTest, PartialStepIsNotAbsorbedIntoTheTarget) {
  const double startTime = 1.0;
  const double ulp = std::nextafter(startTime, 2.0) - startTime;
  const double maxStepSize = 64.0 * ulp;

  for (const double spanInUlps : {65.0, 66.0, 68.0, 128.0, 129.0, 65537.0}) {
    const double targetTime = startTime + spanInUlps * ulp;
    const auto trace = walkSchedule({startTime, targetTime}, maxStepSize);

    EXPECT_TRUE(trace.landedOnEveryPoint) << "span " << spanInUlps << " ulps";
    EXPECT_EQ(trace.endTime, targetTime) << "span " << spanInUlps << " ulps";
    EXPECT_LE(trace.maxStep,
              largestAllowedSubStep({startTime, targetTime}, maxStepSize))
        << "span " << spanInUlps << " ulps";
  }
}

TEST_F(RungeKuttaIntegratorTest, IntegrateLandsExactlyOnSchedulePoints) {
  constexpr std::size_t numIntervals = 99;
  const double maxStepSize = 1.0 / numIntervals;
  integrator_ = std::make_unique<cudaq::integrators::runge_kutta>(
      /*order=*/1, maxStepSize);

  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  cudaq::sum_op<cudaq::matrix_handler> ham(cudaq::spin_op::x(0));
  SystemDynamics system(dims, ham);

  auto initialState = cudaq::state::from_data(initialStateVec);
  auto *castSimState = dynamic_cast<CuDensityMatState *>(
      cudaq::state_helper::getSimulationState(&initialState));
  ASSERT_NE(castSimState, nullptr);
  castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);
  integrator_->setState(initialState, 0.0);

  std::vector<std::complex<double>> steps;
  for (std::size_t i = 0; i <= numIntervals; ++i)
    steps.emplace_back(static_cast<double>(i) / numIntervals, 0.0);
  cudaq::schedule schedule(
      steps, {"t"}, [](const std::string &, const std::complex<double> &value) {
        return value;
      });
  cudaq::integrator_helper::init_system_dynamics(*integrator_, system,
                                                 schedule);

  for (std::size_t i = 1; i < steps.size(); ++i) {
    const double targetTime = steps[i].real();
    integrator_->integrate(targetTime);
    EXPECT_EQ(integrator_->getState().first, targetTime);
  }
}

TEST_F(RungeKuttaIntegratorTest, CheckEvolve) {
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  auto spin_op_x = cudaq::spin_op::x(0);
  cudaq::product_op<cudaq::matrix_handler> ham1 = 2.0 * M_PI * 0.1 * spin_op_x;
  cudaq::sum_op<cudaq::matrix_handler> ham(ham1);
  SystemDynamics system(dims, ham);

  for (int integratorOrder : {1, 2, 4}) {
    std::cout << "Test RK order " << integratorOrder << "\n";
    cudaq::integrators::runge_kutta integrator(integratorOrder, 0.001);
    constexpr std::size_t numDataPoints = 10;
    double t = 0.0;
    auto initialState = cudaq::state::from_data(initialStateVec);
    // initialState.dump();
    auto *simState = cudaq::state_helper::getSimulationState(&initialState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    EXPECT_TRUE(castSimState != nullptr);
    castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);
    integrator.setState(initialState, 0.0);
    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0.0, 1.0 * numDataPoints, numDataPoints)) {
      steps.emplace_back(t, 0.0);
    }
    cudaq::schedule schedule(
        steps, {"t"}, [](const std::string &, const std::complex<double> &val) {
          return val;
        });
    cudaq::integrator_helper::init_system_dynamics(integrator, system,
                                                   schedule);
    std::vector<std::complex<double>> outputStateVec(2);
    for (std::size_t i = 1; i < numDataPoints; ++i) {
      integrator.integrate(i);
      auto [t, state] = integrator.getState();
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

// Test to verify the convergence order of integrators.
// This test uses Richardson extrapolation to estimate the order of accuracy.
// For an integrator of order p, when step size h is halved, error should
// decrease by a factor of 2^p (ratio ~ 2^p).
TEST_F(RungeKuttaIntegratorTest, ConvergenceOrderVerification) {
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  // Hamiltonian: H = omega * sigma_x, omega = 2*pi*0.1
  const double omega = 2.0 * M_PI * 0.1;
  auto spin_op_x = cudaq::spin_op::x(0);
  cudaq::product_op<cudaq::matrix_handler> ham1 = omega * spin_op_x;
  cudaq::sum_op<cudaq::matrix_handler> ham(ham1);
  SystemDynamics system(dims, ham);

  // Test parameters
  const double t_final = 5.0;
  constexpr std::size_t numDataPoints = 51;

  // Helper lambda to run evolution and get final state error
  auto runEvolution = [&](int order, double stepSize) -> double {
    cudaq::integrators::runge_kutta integrator(order, stepSize);
    auto initialState = cudaq::state::from_data(initialStateVec);
    auto *simState = cudaq::state_helper::getSimulationState(&initialState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);
    integrator.setState(initialState, 0.0);

    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0.0, t_final, numDataPoints)) {
      steps.emplace_back(t, 0.0);
    }
    cudaq::schedule schedule(
        steps, {"t"}, [](const std::string &, const std::complex<double> &val) {
          return val;
        });
    cudaq::integrator_helper::init_system_dynamics(integrator, system,
                                                   schedule);

    // Integrate to final time
    integrator.integrate(t_final);
    auto [t, state] = integrator.getState();

    std::vector<std::complex<double>> outputStateVec(2);
    state.to_host(outputStateVec.data(), outputStateVec.size());

    // Analytical solution: |<0|psi>|^2 = cos^2(omega * t)
    double analytical = std::cos(omega * t_final) * std::cos(omega * t_final);
    double numerical = std::norm(outputStateVec[0]); // |<0|psi>|^2 = |psi_0|^2

    return std::abs(numerical - analytical);
  };

  // Test each integrator order with two step sizes
  // Using step sizes that are large enough to show meaningful errors
  const double h1 = 0.1;  // Larger step size
  const double h2 = 0.05; // Half of h1

  for (int order : {1, 2, 4}) {
    double error_h1 = runEvolution(order, h1);
    double error_h2 = runEvolution(order, h2);

    // Compute the error ratio when step size is halved
    double ratio = error_h1 / error_h2;

    // Estimate the convergence order: ratio = 2^p => p = log2(ratio)
    double estimated_order = std::log2(ratio);

    std::cout << "Order " << order << " integrator: "
              << "error(h=" << h1 << ")=" << error_h1 << ", error(h=" << h2
              << ")=" << error_h2 << ", ratio=" << ratio
              << ", estimated_order=" << estimated_order << "\n";

    // Verify that the estimated order is close to the expected order
    // Allow some tolerance due to numerical effects
    // Expected: order 1 -> ratio ~2, order 2 -> ratio ~4, order 4 -> ratio ~16
    double expected_ratio = std::pow(2.0, order);
    double min_acceptable_ratio =
        expected_ratio * 0.5; // At least half of expected

    EXPECT_GE(ratio, min_acceptable_ratio)
        << "Order " << order
        << " integrator shows lower than expected convergence rate. "
        << "Expected ratio >= " << min_acceptable_ratio << ", got " << ratio
        << ". Estimated order: " << estimated_order;

    // Also verify the estimated order is reasonable (within 0.5 of expected)
    EXPECT_GE(estimated_order, order - 0.5)
        << "Order " << order
        << " integrator estimated order too low: " << estimated_order
        << " (expected >= " << (order - 0.5) << ")";
  }
}

// Test that higher-order integrators produce more accurate results
// with the same step size
TEST_F(RungeKuttaIntegratorTest, AccuracyComparison) {
  const std::vector<std::complex<double>> initialStateVec = {{1.0, 0.0},
                                                             {0.0, 0.0}};
  const std::vector<int64_t> dims = {2};
  const double omega = 2.0 * M_PI * 0.1;
  auto spin_op_x = cudaq::spin_op::x(0);
  cudaq::product_op<cudaq::matrix_handler> ham1 = omega * spin_op_x;
  cudaq::sum_op<cudaq::matrix_handler> ham(ham1);
  SystemDynamics system(dims, ham);

  const double t_final = 5.0;
  const double stepSize = 0.1; // Use same step size for all
  constexpr std::size_t numDataPoints = 51;

  // Helper to compute error
  auto computeError = [&](int order) -> double {
    cudaq::integrators::runge_kutta integrator(order, stepSize);
    auto initialState = cudaq::state::from_data(initialStateVec);
    auto *simState = cudaq::state_helper::getSimulationState(&initialState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    castSimState->initialize_cudm(handle_, dims, /*batchSize=*/1);
    integrator.setState(initialState, 0.0);

    std::vector<std::complex<double>> steps;
    for (double t : cudaq::linspace(0.0, t_final, numDataPoints)) {
      steps.emplace_back(t, 0.0);
    }
    cudaq::schedule schedule(
        steps, {"t"}, [](const std::string &, const std::complex<double> &val) {
          return val;
        });
    cudaq::integrator_helper::init_system_dynamics(integrator, system,
                                                   schedule);

    integrator.integrate(t_final);
    auto [t, state] = integrator.getState();

    std::vector<std::complex<double>> outputStateVec(2);
    state.to_host(outputStateVec.data(), outputStateVec.size());

    double analytical = std::cos(omega * t_final) * std::cos(omega * t_final);
    double numerical = std::norm(outputStateVec[0]);

    return std::abs(numerical - analytical);
  };

  double error_order1 = computeError(1);
  double error_order2 = computeError(2);
  double error_order4 = computeError(4);

  std::cout << "Accuracy comparison (step_size=" << stepSize << "):\n";
  std::cout << "  Order 1 error: " << error_order1 << "\n";
  std::cout << "  Order 2 error: " << error_order2 << "\n";
  std::cout << "  Order 4 error: " << error_order4 << "\n";

  // Order 2 should be significantly more accurate than Order 1
  EXPECT_LT(error_order2, error_order1 * 0.1)
      << "Order 2 should be at least 10x more accurate than Order 1";

  // Order 4 should be significantly more accurate than Order 2
  EXPECT_LT(error_order4, error_order2 * 0.01)
      << "Order 4 should be at least 100x more accurate than Order 2";

  // Order 2 error should be reasonable for a 2nd order method
  // With h=0.1 and t=5.0 (50 steps), error should be O(h^2) ~ 0.01 range
  EXPECT_LT(error_order2, 0.01)
      << "Order 2 error too large for a proper 2nd-order method";
}
