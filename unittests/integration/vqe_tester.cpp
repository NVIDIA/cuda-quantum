/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <cudaq/builder.h>

#include <cudaq/algorithms/gradients/central_difference.h>
#include <cudaq/optimizers.h>

#ifndef CUDAQ_BACKEND_DM

struct ansatz_compute_action {
  void operator()(std::vector<double> theta) __qpu__ {
    cudaq::qvector q(4);
    x(q[0]);
    x(q[2]);

    cudaq::compute_action(
        [&]() {
          rx(M_PI_2, q[0]);
          h(q[1]);
          h(q[2]);
          h(q[3]);
          x<cudaq::ctrl>(q[0], q[1]);
          x<cudaq::ctrl>(q[1], q[2]);
          x<cudaq::ctrl>(q[2], q[3]);
        },
        [&]() { rz(theta[0], q[3]); });
  }
};

struct ansatz_compute_action_double {
  void operator()(double theta) __qpu__ {
    cudaq::qvector q(4);
    x(q[0]);
    x(q[2]);

    // above is equivalent to this
    cudaq::compute_action(
        [&]() {
          rx(M_PI_2, q[0]);
          h(q[1]);
          h(q[2]);
          h(q[3]);
          x<cudaq::ctrl>(q[0], q[1]);
          x<cudaq::ctrl>(q[1], q[2]);
          x<cudaq::ctrl>(q[2], q[3]);
        },
        [&]() { rz(theta, q[3]); });
  }
};

class VQETester : public testing::Test {
protected:
  // Per-test-suite set-up.
  // Called before the first test in this test suite.
  // Can be omitted if not needed.
  static void SetUpTestSuite() {
    std::vector<double> h2_data{0, 0, 0, 0, -0.10647701149499994, 0.0,
                                1, 1, 1, 1, 0.0454063328691,      0.0,
                                1, 1, 3, 3, 0.0454063328691,      0.0,
                                3, 3, 1, 1, 0.0454063328691,      0.0,
                                3, 3, 3, 3, 0.0454063328691,      0.0,
                                2, 0, 0, 0, 0.170280101353,       0.0,
                                2, 2, 0, 0, 0.120200490713,       0.0,
                                2, 0, 2, 0, 0.168335986252,       0.0,
                                2, 0, 0, 2, 0.165606823582,       0.0,
                                0, 2, 0, 0, -0.22004130022499996, 0.0,
                                0, 2, 2, 0, 0.165606823582,       0.0,
                                0, 2, 0, 2, 0.174072892497,       0.0,
                                0, 0, 2, 0, 0.17028010135300004,  0.0,
                                0, 0, 2, 2, 0.120200490713,       0.0,
                                0, 0, 0, 2, -0.22004130022499999, 0.0,
                                15};
    // auto array = __quantum__rt__array_create_1d(sizeof(double),
    // h2_data.size()); for (std::size_t i = 0; i < h2_data.size(); i++) {
    //   int8_t *raw = __quantum__rt__array_get_element_ptr_1d(array, i);
    //   auto ptr = reinterpret_cast<double *>(raw);
    //   (*ptr) = h2_data[i];
    // }
    H = std::make_unique<cudaq::spin_op>(h2_data, 4);
  }

  template <typename Kernel>
  std::unique_ptr<cudaq::gradient> genGradient() {
    return std::make_unique<cudaq::gradients::central_difference>(Kernel{});
  }
  template <typename Kernel, typename ArgMapper>
  std::unique_ptr<cudaq::gradient> genGradient(ArgMapper &&argm) {
    return std::make_unique<cudaq::gradients::central_difference>(Kernel{},
                                                                  argm);
  }
  // Per-test-suite tear-down.
  // Called after the last test in this test suite.
  // Can be omitted if not needed.
  static void TearDownTestSuite() {}
  static std::unique_ptr<cudaq::spin_op> H;
};

std::unique_ptr<cudaq::spin_op> VQETester::H = nullptr;

CUDAQ_TEST_F(VQETester, checkGradientFree) {
  printf("Default run with std::vector<double>\n");
  cudaq::optimizers::cobyla c_opt;
  auto [opt_val, opt_params] =
      cudaq::vqe(ansatz_compute_action{}, *H, c_opt, 1);
  EXPECT_NEAR(opt_val, -1.1371, 1e-3);
}

CUDAQ_TEST_F(VQETester, checkSpsa) {
  printf("Run with spsa\n");
  cudaq::optimizers::spsa opt;
  opt.alpha = .01;
  auto [opt_val, opt_params] = cudaq::vqe(ansatz_compute_action{}, *H, opt, 1);
  EXPECT_NEAR(opt_val, -1.1371, 1e-3);
}

CUDAQ_TEST_F(VQETester, checkDifferentArgStructure) {
  cudaq::optimizers::cobyla c_opt;
  auto argMapper = [](std::vector<double> x) { return std::make_tuple(x[0]); };
  auto [opt_val2, opt_params2] =
      cudaq::vqe(ansatz_compute_action_double{}, *H, c_opt, 1, argMapper);
  EXPECT_NEAR(opt_val2, -1.1371, 1e-3);

  cudaq::optimizers::lbfgs l_opt;
  auto [opt_val4, opt_params4] =
      cudaq::vqe(ansatz_compute_action_double{},
                 *genGradient<ansatz_compute_action_double>(argMapper), *H,
                 l_opt, 1, argMapper);
  EXPECT_NEAR(opt_val4, -1.1371, 1e-3);
}

CUDAQ_TEST_F(VQETester, checkThrowNoGradient) {
  cudaq::optimizers::lbfgs l_opt;
  EXPECT_ANY_THROW({ cudaq::vqe(ansatz_compute_action{}, *H, l_opt, 1); });
}

CUDAQ_TEST_F(VQETester, checkGradientBased) {
  cudaq::optimizers::lbfgs l_opt;
  auto [opt_val2, opt_params2] =
      cudaq::vqe(ansatz_compute_action{}, *genGradient<ansatz_compute_action>(),
                 *H, l_opt, 1);
  EXPECT_NEAR(opt_val2, -1.1371, 1e-3);
}

CUDAQ_TEST_F(VQETester, checkBuilderVqe) {
  cudaq::optimizers::lbfgs l_opt;

  auto [kernel, params] = cudaq::make_kernel<std::vector<double>>();
  auto q = kernel.qalloc(4);

  kernel.x(q[0]);
  kernel.x(q[2]);
  kernel.rx(M_PI_2, q[0]);
  kernel.h(q[1]);
  kernel.h(q[2]);
  kernel.h(q[3]);
  kernel.x<cudaq::ctrl>(q[0], q[1]);
  kernel.x<cudaq::ctrl>(q[1], q[2]);
  kernel.x<cudaq::ctrl>(q[2], q[3]);
  kernel.rz(params[0], q[3]);
  kernel.x<cudaq::ctrl>(q[2], q[3]);
  kernel.x<cudaq::ctrl>(q[1], q[2]);
  kernel.x<cudaq::ctrl>(q[0], q[1]);
  kernel.h(q[3]);
  kernel.h(q[2]);
  kernel.h(q[1]);
  kernel.rx(-M_PI_2, q[0]);

  cudaq::gradients::central_difference gradient(kernel);

  printf("Run with kernel kernel, with lbfgs.\n");
  auto [opt_val3, opt_params3] = cudaq::vqe(kernel, gradient, *H, l_opt, 1);
  EXPECT_NEAR(opt_val3, -1.1371, 1e-3);
}

CUDAQ_TEST_F(VQETester, checkArgMapper) {
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  cudaq::spin_op h3 = h + 9.625 - 9.625 * z(2) - 3.913119 * x(1) * x(2) -
                      3.913119 * y(1) * y(2);

  auto ansatz = [](double theta, double phi) __qpu__ {
    cudaq::qvector q(3);
    x(q[0]);
    ry(theta, q[1]);
    ry(phi, q[2]);
    x<cudaq::ctrl>(q[2], q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    ry(-theta, q[1]);
    x<cudaq::ctrl>(q[0], q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  };

  auto argMapper = [](std::vector<double> x) {
    return std::make_tuple(x[0], x[1]);
  };

  cudaq::gradients::central_difference grad(ansatz, argMapper);
  cudaq::optimizers::lbfgs optimizer;
  // Without the ArgMapper, this will not compile
  auto [opt_val_0, optpp] =
      cudaq::vqe(ansatz, grad, h3, optimizer, 2, argMapper);
  printf("<H3> = %lf\n", opt_val_0);
  EXPECT_NEAR(opt_val_0, -2.045375, 1e-3);
}

CUDAQ_TEST_F(VQETester, checkThrowInvalidRuntimeArgs) {

  // Build up a 2 parameter circuit using a vector<double> parameter
  // Run the cudaq optimizer to find optimal value.
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  cudaq::spin_op h3 = h + 9.625 - 9.625 * z(2) - 3.913119 * x(1) * x(2) -
                      3.913119 * y(1) * y(2);

  auto [ansatz, theta, phi] = cudaq::make_kernel<double, double>();

  auto q = ansatz.qalloc(3);
  ansatz.x(q[0]);
  ansatz.ry(theta, q[1]);
  ansatz.ry(phi, q[2]);
  ansatz.x<cudaq::ctrl>(q[2], q[0]);
  ansatz.x<cudaq::ctrl>(q[0], q[1]);
  ansatz.ry(-theta, q[1]);
  ansatz.x<cudaq::ctrl>(q[0], q[1]);
  ansatz.x<cudaq::ctrl>(q[1], q[0]);

  auto argMapper = [](std::vector<double> x) {
    return std::make_tuple(x[0], x[1]);
  };
  cudaq::gradients::central_difference grad(ansatz, argMapper);
  cudaq::optimizers::lbfgs optimizer;

  // Correct usage...
  auto [opt_val_0, xx] = cudaq::vqe(ansatz, grad, h3, optimizer, 2, argMapper);
  printf("<H3> = %lf\n", opt_val_0);
  EXPECT_NEAR(opt_val_0, -2.045375, 1e-3);
}

#endif
