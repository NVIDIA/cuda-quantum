/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <cudaq/algorithms/gradients/central_difference.h>
#include <cudaq/builder.h>
#include <cudaq/optimizers.h>
#include <regex>

CUDAQ_TEST(BuilderTester, checkSimple) {
  {
    using namespace cudaq::spin;
    cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                       .21829 * z(0) - 6.125 * z(1);

    auto [ansatz, theta] = cudaq::make_kernel<double>();

    // // Allocate some qubits
    auto q = ansatz.qalloc(2);

    // Build up the circuit, use the acquired parameter
    ansatz.x(q[0]);
    ansatz.ry(theta, q[1]);
    ansatz.x<cudaq::ctrl>(q[1], q[0]);

    // Create the kernel, can be passed to cudaq algorithms
    // just like a declared kernel type. Instantiate
    // invalidates the qvector reference you have.
    double exp = cudaq::observe(ansatz, h, .59);
    printf("<H2> = %lf\n", exp);
    EXPECT_NEAR(exp, -1.748795, 1e-2);
  }

  {
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
    cudaq::gradients::central_difference gradient(ansatz, argMapper);
    cudaq::optimizers::lbfgs optimizer;
    optimizer.initial_parameters = {0.359, 0.257};
    optimizer.max_eval = 4;
    optimizer.max_line_search_trials = 8;
    auto [opt_val_0, optpp] =
        cudaq::vqe(ansatz, gradient, h3, optimizer, 2, argMapper);
    printf("Opt-params: %lf %lf \n", optpp[0], optpp[1]);
    printf("<H3> = %lf\n", opt_val_0);
    EXPECT_NEAR(opt_val_0, -2.045375, 1e-2);
  }

  {
    // Build up a 2 parameter circuit using a vector<double> parameter
    // Run the cudaq optimizer to find optimal value.
    using namespace cudaq::spin;
    cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                       .21829 * z(0) - 6.125 * z(1);
    cudaq::spin_op h3 = h + 9.625 - 9.625 * z(2) - 3.913119 * x(1) * x(2) -
                        3.913119 * y(1) * y(2);

    auto [ansatz, thetas] = cudaq::make_kernel<std::vector<double>>();

    auto q = ansatz.qalloc(3);

    ansatz.x(q[0]);
    ansatz.ry(thetas[0], q[1]);
    ansatz.ry(thetas[1], q[2]);
    ansatz.x<cudaq::ctrl>(q[2], q[0]);
    ansatz.x<cudaq::ctrl>(q[0], q[1]);
    ansatz.ry(-thetas[0], q[1]);
    ansatz.x<cudaq::ctrl>(q[0], q[1]);
    ansatz.x<cudaq::ctrl>(q[1], q[0]);

    cudaq::gradients::central_difference gradient(ansatz);
    cudaq::optimizers::lbfgs optimizer;
    optimizer.initial_parameters = {0.359, 0.257};
    optimizer.max_eval = 4;
    optimizer.max_line_search_trials = 8;
    auto [opt_val_0, optpp] = cudaq::vqe(ansatz, gradient, h3, optimizer, 2);
    printf("<H3> = %lf\n", opt_val_0);
    EXPECT_NEAR(opt_val_0, -2.045375, 1e-2);
  }

  {
    int n_qubits = 4;
    auto ghz_builder = cudaq::make_kernel();
    auto q = ghz_builder.qalloc(n_qubits);
    ghz_builder.h(q[0]);
    for (int i = 0; i < n_qubits - 1; i++) {
      ghz_builder.x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    ghz_builder.mz(q);

    auto counts = cudaq::sample(ghz_builder);
    counts.dump();
    int counter = 0;
    for (auto &[k, v] : counts) {
      counter += v;
      EXPECT_TRUE(k == "0000" || k == "1111");
    }
    EXPECT_EQ(counter, 1000);
  }

  {
    auto ccnot_builder = cudaq::make_kernel();
    auto q = ccnot_builder.qalloc(3);
    ccnot_builder.x(q);
    ccnot_builder.x(q[1]);
    ccnot_builder.x<cudaq::ctrl>(q[0], q[1], q[2]);
    ccnot_builder.mz(q);

    auto counts = cudaq::sample(ccnot_builder);
    counts.dump();
    EXPECT_TRUE(counts.begin()->first == "101");
  }

  {
    // Check controlled parametric gates (constant angle)
    auto cnot_builder = cudaq::make_kernel();
    auto q = cnot_builder.qalloc(2);
    cnot_builder.x(q);
    // Rx(pi) == X
    cnot_builder.rx<cudaq::ctrl>(M_PI, q[0], q[1]);
    cnot_builder.mz(q);

    auto counts = cudaq::sample(cnot_builder);
    counts.dump();
    EXPECT_EQ(counts.size(), 1);
    EXPECT_TRUE(counts.begin()->first == "10");
  }

  {
    // Check controlled parametric gates (QuakeValue angle)
    auto [cnot_builder, theta] = cudaq::make_kernel<double>();
    auto q = cnot_builder.qalloc(2);
    cnot_builder.x(q);
    // controlled-Rx(theta)
    cnot_builder.rx<cudaq::ctrl>(theta, q[0], q[1]);
    cnot_builder.mz(q);
    // assign theta = pi; controlled-Rx(pi) == CNOT
    auto counts = cudaq::sample(cnot_builder, M_PI);
    counts.dump();
    EXPECT_EQ(counts.size(), 1);
    EXPECT_TRUE(counts.begin()->first == "10");
  }

  {
    // Check adjoint parametric gates (constant angles)
    auto rx_builder = cudaq::make_kernel();
    auto q = rx_builder.qalloc();
    // Rx(pi) == X
    rx_builder.rx<cudaq::adj>(-M_PI_2, q);
    rx_builder.rx(M_PI_2, q);
    rx_builder.mz(q);

    auto counts = cudaq::sample(rx_builder);
    counts.dump();
    EXPECT_EQ(counts.size(), 1);
    EXPECT_TRUE(counts.begin()->first == "1");
  }

  {
    // Check adjoint parametric gates (constant angles, implicit type
    // conversion)
    auto rx_builder = cudaq::make_kernel();
    auto q = rx_builder.qalloc();
    // float -> double implicit type conversion
    rx_builder.rx<cudaq::adj>(-M_PI_4, q);
    rx_builder.rx(M_PI_4, q);
    // long double -> double implicit type conversion
    const long double pi_4_ld = M_PI / 4.0;
    rx_builder.rx<cudaq::adj>(-pi_4_ld, q);
    rx_builder.rx(pi_4_ld, q);
    rx_builder.mz(q);
    // Rx(pi) == X (four pi/4 rotations)
    auto counts = cudaq::sample(rx_builder);
    counts.dump();
    EXPECT_EQ(counts.size(), 1);
    EXPECT_TRUE(counts.begin()->first == "1");
  }

  {
    // Check adjoint parametric gates (QuakeValue angle)
    auto [rx_builder, angle] = cudaq::make_kernel<double>();
    auto q = rx_builder.qalloc();
    rx_builder.rx<cudaq::adj>(-angle, q);
    rx_builder.rx(angle, q);
    rx_builder.mz(q);
    // angle = pi/2 => equivalent to Rx(pi) == X
    auto counts = cudaq::sample(rx_builder, M_PI_2);
    counts.dump();
    EXPECT_EQ(counts.size(), 1);
    EXPECT_TRUE(counts.begin()->first == "1");
  }
}

CUDAQ_TEST(BuilderTester, checkRotations) {

  // rx: entire qvector
  {
    cudaq::set_random_seed(4);

    auto kernel = cudaq::make_kernel();
    auto targets = kernel.qalloc(3);
    auto extra = kernel.qalloc();

    // Rotate only our target qubits to |1> along X.
    kernel.rx(M_PI, targets);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    EXPECT_EQ(counts.count("1110"), 1000);
  }

  // ry: entire qvector
  {
    cudaq::set_random_seed(4);

    auto kernel = cudaq::make_kernel();
    auto targets = kernel.qalloc(3);
    auto extra = kernel.qalloc();

    // Rotate only our target qubits to |1> along Y.
    kernel.ry(M_PI, targets);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    EXPECT_EQ(counts.count("1110"), 1000);
  }

  // rz: entire qvector
  {
    cudaq::set_random_seed(4);

    auto kernel = cudaq::make_kernel();
    auto targets = kernel.qalloc(3);
    auto extra = kernel.qalloc();

    // Place targets in superposition state.
    kernel.h(targets);
    // Rotate our targets around Z by -pi.
    kernel.rz(-M_PI, targets);
    kernel.h(targets);

    // All targets should be in the |1> state.
    auto counts = cudaq::sample(kernel);
    counts.dump();
    EXPECT_EQ(counts.count("1110"), 1000);
  }

  // r1: entire qvector
  {
    cudaq::set_random_seed(4);

    auto kernel = cudaq::make_kernel();
    auto targets = kernel.qalloc(3);
    auto extra = kernel.qalloc();

    kernel.x(targets);
    kernel.h(targets);
    // Rotate our targets around Z by -pi.
    kernel.r1(-M_PI, targets);
    kernel.h(targets);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    // Qubits should all be back in |0>.
    EXPECT_EQ(counts.count("0000"), 1000);
  }

  // controlled-rx
  {
    auto [kernel, val] = cudaq::make_kernel<float>();
    auto target = kernel.qalloc();
    auto q1 = kernel.qalloc();
    auto q2 = kernel.qalloc();
    auto q3 = kernel.qalloc();

    // Prepare control qubits in the 1-state.
    kernel.x(q1);
    kernel.x(q2);
    kernel.x(q3);

    // Create a vector of controls.
    std::vector<cudaq::QuakeValue> ctrls{q1, q2, q3};

    // Overload 1: `QuakeValue` parameter.
    kernel.rx<cudaq::ctrl>(val, ctrls, target);
    // Overload 2: `double` parameter.
    kernel.rx<cudaq::ctrl>(M_PI, ctrls, target);

    auto counts = cudaq::sample(kernel, M_PI);
    counts.dump();

    // Our controls should remain in the 1-state, while
    // the target has been rotated by `2*M_PI = 2pi`. I.e, identity.
    EXPECT_EQ(counts.count("0111"), 1000);
  }

  // controlled-ry
  {
    auto [kernel, val] = cudaq::make_kernel<float>();
    auto target = kernel.qalloc();
    auto q1 = kernel.qalloc();
    auto q2 = kernel.qalloc();
    auto q3 = kernel.qalloc();

    // Prepare control qubits in the 1-state.
    kernel.x(q1);
    kernel.x(q2);
    kernel.x(q3);

    // Create a vector of controls.
    std::vector<cudaq::QuakeValue> ctrls{q1, q2, q3};

    // Overload 1: `QuakeValue` parameter.
    kernel.rx<cudaq::ctrl>(val, ctrls, target);
    // Overload 2: `double` parameter.
    kernel.rx<cudaq::ctrl>(M_PI, ctrls, target);

    auto counts = cudaq::sample(kernel, M_PI);
    counts.dump();

    // Our controls should remain in the 1-state, while
    // the target has been rotated by `2*M_PI = 2pi`. I.e, identity.
    EXPECT_EQ(counts.count("0111"), 1000);
  }

  // controlled-rz
  {
    auto [kernel, val] = cudaq::make_kernel<float>();
    auto target = kernel.qalloc();
    auto q1 = kernel.qalloc();
    auto q2 = kernel.qalloc();
    auto q3 = kernel.qalloc();

    // Prepare control qubits in the 1-state.
    kernel.x(q1);
    kernel.x(q2);
    kernel.x(q3);

    // X + Hadamard on target qubit.
    kernel.x(target);
    kernel.h(target);

    // Create a vector of controls.
    std::vector<cudaq::QuakeValue> ctrls{q1, q2, q3};

    // Overload 1: `QuakeValue` parameter.
    kernel.rz<cudaq::ctrl>(val, ctrls, target);
    // Overload 2: `double` parameter.
    kernel.rz<cudaq::ctrl>(-M_PI_2, ctrls, target);

    // Hadamard the target again.
    kernel.h(target);

    auto counts = cudaq::sample(kernel, -M_PI_2);
    counts.dump();

    // The phase rotations on our target by a total of -pi should
    // return it to the 0-state.
    EXPECT_EQ(counts.count("0111"), 1000);
  }

  // controlled-r1
  {
    auto [kernel, val] = cudaq::make_kernel<float>();
    auto target = kernel.qalloc();
    auto q1 = kernel.qalloc();
    auto q2 = kernel.qalloc();
    auto q3 = kernel.qalloc();

    // Prepare control qubits in the 1-state.
    kernel.x(q1);
    kernel.x(q2);
    kernel.x(q3);

    // X + Hadamard on target qubit.
    kernel.x(target);
    kernel.h(target);

    // Create a vector of controls.
    std::vector<cudaq::QuakeValue> ctrls{q1, q2, q3};

    // Overload 1: `QuakeValue` parameter.
    kernel.r1<cudaq::ctrl>(val, ctrls, target);
    // Overload 2: `double` parameter.
    kernel.r1<cudaq::ctrl>(-M_PI_2, ctrls, target);

    // Hadamard the target again.
    kernel.h(target);

    auto counts = cudaq::sample(kernel, -M_PI_2);
    counts.dump();

    // The phase rotations on our target by a total of -pi should
    // return it to the 0-state.
    EXPECT_EQ(counts.count("0111"), 1000);
  }
}

CUDAQ_TEST(BuilderTester, checkSwap) {
  cudaq::set_random_seed(13);

  // Simple two-qubit swap.
  {
    auto kernel = cudaq::make_kernel();
    auto first = kernel.qalloc();
    auto second = kernel.qalloc();
    // `first` qubit into the 1-state.
    kernel.x(first);
    // Swap their states and measure.
    kernel.swap(first, second);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    EXPECT_NEAR(counts.count("01"), 1000, 0);
  }

  // Simple two-qubit swap.
  {
    auto kernel = cudaq::make_kernel();
    auto first = kernel.qalloc();
    auto second = kernel.qalloc();
    // `second` qubit into the 1-state.
    kernel.x(second);
    // Swap their states and measure.
    kernel.swap(first, second);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    EXPECT_NEAR(counts.count("10"), 1000, 0);
  }

  // Single qubit controlled-SWAP.
  {
    auto kernel = cudaq::make_kernel();
    auto ctrl = kernel.qalloc();
    auto first = kernel.qalloc();
    auto second = kernel.qalloc();
    // ctrl and `first` in the 1-state.
    kernel.x(ctrl);
    kernel.x(first);
    // Swap their states and measure.
    kernel.swap<cudaq::ctrl>(ctrl, first, second);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    EXPECT_NEAR(counts.count("101"), 1000, 0);
  }

  // Multi-controlled SWAP with a ctrl register.
  {
    auto kernel = cudaq::make_kernel();
    auto ctrls = kernel.qalloc(3);
    auto first = kernel.qalloc();
    auto second = kernel.qalloc();

    // Rotate `first` to |1> state.
    kernel.x(first);

    // Only a subset of controls in the |1> state.
    kernel.x(ctrls[0]);
    // No SWAP should occur.
    kernel.swap<cudaq::ctrl>(ctrls, first, second);

    // Flip the rest of the controls to |1>.
    kernel.x(ctrls[1]);
    kernel.x(ctrls[2]);
    // `first` and `second` should SWAP.
    kernel.swap<cudaq::ctrl>(ctrls, first, second);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    std::string ctrls_state = "111";
    // `first` is now |0>, `second` is now |1>.
    std::string want_target = "01";
    auto want_state = ctrls_state + want_target;
    EXPECT_NEAR(counts.count(want_state), 1000, 0);
  }

  // Multi-controlled SWAP with a vector of ctrl qubits.
  {
    auto kernel = cudaq::make_kernel();
    std::vector<cudaq::QuakeValue> ctrls{kernel.qalloc(), kernel.qalloc(),
                                         kernel.qalloc()};
    auto first = kernel.qalloc();
    auto second = kernel.qalloc();

    // Rotate `second` to |1> state.
    kernel.x(second);

    // Only a subset of controls in the |1> state.
    kernel.x(ctrls[0]);
    // No SWAP should occur.
    kernel.swap<cudaq::ctrl>(ctrls, first, second);

    // Flip the rest of the controls to |1>.
    kernel.x(ctrls[1]);
    kernel.x(ctrls[2]);
    // `first` and `second` should SWAP.
    kernel.swap<cudaq::ctrl>(ctrls, first, second);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    std::string ctrls_state = "111";
    // `first` is now |1>, `second` is now |0>.
    std::string want_target = "10";
    auto want_state = ctrls_state + want_target;
    EXPECT_NEAR(counts.count(want_state), 1000, 0);
  }

  // Multi-controlled SWAP with a variadic list of ctrl qubits.
  {
    auto kernel = cudaq::make_kernel();
    auto ctrls0 = kernel.qalloc(2);
    auto ctrls1 = kernel.qalloc();
    auto ctrls2 = kernel.qalloc(2);
    auto first = kernel.qalloc();
    auto second = kernel.qalloc();

    // Rotate `second` to |1> state.
    kernel.x(second);

    // Only a subset of controls in the |1> state.
    kernel.x(ctrls0);
    // No SWAP should occur.
    kernel.swap<cudaq::ctrl>(ctrls0, ctrls1, ctrls2, first, second);

    // Flip the rest of the controls to |1>.
    kernel.x(ctrls1);
    kernel.x(ctrls2);
    // `first` and `second` should SWAP.
    kernel.swap<cudaq::ctrl>(ctrls0, ctrls1, ctrls2, first, second);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    std::string ctrls_state = "11111";
    // `first` is now |1>, `second` is now |0>.
    std::string want_target = "10";
    auto want_state = ctrls_state + want_target;
    EXPECT_NEAR(counts.count(want_state), 1000, 0);
  }
}

// Conditional execution on the tensornet backend is slow for a large number of
// shots.
#ifndef CUDAQ_BACKEND_TENSORNET
CUDAQ_TEST(BuilderTester, checkConditional) {
  {
    cudaq::set_random_seed(13);
    auto kernel = cudaq::make_kernel();
    auto q = kernel.qalloc(2);
    kernel.h(q[0]);
    auto mres = kernel.mz(q[0], "res0");
    kernel.c_if(mres, [&]() { kernel.x(q[1]); });
    kernel.mz(q);

    printf("%s\n", kernel.to_quake().c_str());

    auto counts = cudaq::sample(kernel);
    counts.dump();
    EXPECT_EQ(counts.register_names().size(), 2);
    EXPECT_EQ(counts.size("res0"), 2);
    EXPECT_NEAR(counts.count("11") / 1000., 0.5, 1e-1);
    EXPECT_NEAR(counts.count("00") / 1000., 0.5, 1e-1);
    EXPECT_NEAR(counts.count("1", "res0") / 1000., 0.5, 1e-1);
    EXPECT_NEAR(counts.count("0", "res0") / 1000., 0.5, 1e-1);
  }

  //  Tests a previous bug where the `extract_ref` for a qubit
  //  would get hidden within a conditional. This would result in
  //  the runtime error "operator #0 does not dominate this use".
  {
    auto kernel = cudaq::make_kernel();
    auto qreg = kernel.qalloc(3);

    kernel.x(qreg[1]);
    auto measure0 = kernel.mz(qreg[1]);

    kernel.c_if(measure0, [&]() { kernel.x(qreg[0]); });

    // Now we try to use `qreg[0]` again.
    kernel.x(qreg[0]);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    EXPECT_EQ(counts.count("010"), 1000);
  }
}
#endif

CUDAQ_TEST(BuilderTester, checkQubitArg) {
  auto [kernel, qubitArg] = cudaq::make_kernel<cudaq::qubit>();
  kernel.h(qubitArg);

  printf("%s", kernel.to_quake().c_str());

  auto entryPoint = cudaq::make_kernel();
  auto qubit = entryPoint.qalloc();
  entryPoint.call(kernel, qubit);
  entryPoint.mz(qubit);

  printf("%s", entryPoint.to_quake().c_str());

  auto counts = cudaq::sample(entryPoint);
  counts.dump();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(BuilderTester, checkQvecArg) {
  auto [kernel, qvectorArg] = cudaq::make_kernel<cudaq::qvector<>>();
  kernel.h(qvectorArg);

  printf("%s", kernel.to_quake().c_str());

  auto entryPoint = cudaq::make_kernel();
  auto qubits = entryPoint.qalloc(5);
  entryPoint.call(kernel, qubits);
  entryPoint.mz(qubits);

  printf("%s", entryPoint.to_quake().c_str());

  auto counts = cudaq::sample(entryPoint);
  counts.dump();
  EXPECT_EQ(counts.size(), 32);
  EXPECT_EQ(counts.to_map().begin()->first.length(), 5);
}

CUDAQ_TEST(BuilderTester, checkSlice) {
  auto [kernel, params] = cudaq::make_kernel<std::vector<double>>();
  auto q = kernel.qalloc(4);
  auto qSliced = q.slice(0, 2);
  auto sliced = params.slice(0, 2);
  kernel.rx(sliced[0], qSliced[0]);
  kernel.ry(sliced[1], qSliced[0]);
  kernel.x<cudaq::ctrl>(qSliced[0], qSliced[1]);
  kernel.mz(qSliced);

  std::cout << kernel.to_quake() << "\n";

  // Correct number of params provided
  kernel(std::vector<double>{M_PI, M_PI_2});

  // Not enough provided, we should have at least 2
  EXPECT_ANY_THROW({ kernel(std::vector<double>{M_PI}); });

  auto test = cudaq::make_kernel();
  auto q2 = test.qalloc(2);
  // Should throw since we have 2 qubits and asked for 3
  EXPECT_ANY_THROW({ auto sliced = q2.slice(0, 3); });
}

CUDAQ_TEST(BuilderTester, checkStdVecValidate) {
  auto [kernel, thetas] = cudaq::make_kernel<std::vector<double>>();
  auto q = kernel.qalloc(2);
  kernel.rx(thetas[0], q[0]);
  kernel.ry(thetas[1], q[0]);
  kernel.x<cudaq::ctrl>(q[0], q[1]);
  kernel.mz(q);

  // This is ok
  kernel(std::vector<double>{M_PI, M_PI_2});

  // This is not ok
  EXPECT_ANY_THROW({ kernel(std::vector<double>{M_PI}); });
}

CUDAQ_TEST(BuilderTester, checkIsArgStdVec) {
  auto [kernel, one, two, thetas, four] =
      cudaq::make_kernel<double, float, std::vector<double>, int>();

  EXPECT_TRUE(kernel.isArgStdVec(2));
  EXPECT_FALSE(kernel.isArgStdVec(1));
}

CUDAQ_TEST(BuilderTester, checkKernelControl) {
  cudaq::set_random_seed(13);

  auto [xPrep, qubitIn] = cudaq::make_kernel<cudaq::qubit>();
  xPrep.x(qubitIn);

  auto [hPrep, qubitIn2] = cudaq::make_kernel<cudaq::qubit>();
  hPrep.h(qubitIn2);

  // Compute <1|X|1> = 0
  auto hadamardTest = cudaq::make_kernel();
  auto q = hadamardTest.qalloc();
  auto ancilla = hadamardTest.qalloc();
  hadamardTest.call(xPrep, q);
  hadamardTest.h(ancilla);
  hadamardTest.control(xPrep, ancilla, q);
  hadamardTest.h(ancilla);
  hadamardTest.mz(ancilla);

  printf("%s\n", hadamardTest.to_quake().c_str());
  auto counts = cudaq::sample(10000, hadamardTest);
  counts.dump();
  printf("< 1 | X | 1 > = %lf\n", counts.expectation());
  EXPECT_NEAR(counts.expectation(), 0.0, 1e-1);

  // Compute <1|H|1> = 1.
  auto hadamardTest2 = cudaq::make_kernel();
  auto q2 = hadamardTest2.qalloc();
  auto ancilla2 = hadamardTest2.qalloc();
  hadamardTest2.call(xPrep, q2);
  hadamardTest2.h(ancilla2);
  hadamardTest2.control(hPrep, ancilla2, q2);
  hadamardTest2.h(ancilla2);
  hadamardTest2.mz(ancilla2);

  printf("%s\n", hadamardTest2.to_quake().c_str());
  counts = cudaq::sample(10000, hadamardTest2);
  printf("< 1 | H | 1 > = %lf\n", counts.expectation());
  EXPECT_NEAR(counts.expectation(), -1.0 / std::sqrt(2.0), 1e-1);

  // Demonstrate can control on qvector
  auto kernel = cudaq::make_kernel();
  auto ctrls = kernel.qalloc(2);
  auto tgt = kernel.qalloc();
  // Prep 101
  kernel.x(ctrls[0]);
  kernel.x(tgt);
  kernel.control(xPrep, ctrls, tgt);
  kernel.mz(ctrls);
  kernel.mz(tgt);
  printf("%s\n", kernel.to_quake().c_str());
  counts = cudaq::sample(kernel);
  counts.dump();
  EXPECT_EQ(1, counts.size());
  EXPECT_TRUE(counts.begin()->first == "101");
}

CUDAQ_TEST(BuilderTester, checkAdjointOp) {
  auto kernel = cudaq::make_kernel();
  auto q = kernel.qalloc();
  kernel.t<cudaq::adj>(q);
  kernel.t(q);
  kernel.mz(q);
  printf("%s\n", kernel.to_quake().c_str());
  cudaq::sample(kernel).dump();
}

CUDAQ_TEST(BuilderTester, checkAdjointOpRvalQuakeValue) {
  auto kernel = cudaq::make_kernel();
  // allocate more than 1 qubits so that we can use QuakeValue::operator[],
  // which returns an r-val QuakeValue.
  auto qubits = kernel.qalloc(2);
  kernel.h(qubits[0]);
  // T-dagger - T = I
  kernel.t<cudaq::adj>(qubits[0]);
  kernel.t(qubits[0]);
  kernel.h(qubits[0]);
  kernel.mz(qubits[0]);
  printf("%s\n", kernel.to_quake().c_str());
  auto counts = cudaq::sample(kernel);
  counts.dump();
  EXPECT_EQ(1, counts.size());
  EXPECT_TRUE(counts.begin()->first == "0");
}

CUDAQ_TEST(BuilderTester, checkKernelAdjoint) {
  auto [kernel, qubit] = cudaq::make_kernel<cudaq::qubit>();
  kernel.h(qubit);
  kernel.t(qubit);
  kernel.s(qubit);
  kernel.t<cudaq::adj>(qubit);

  auto entryPoint = cudaq::make_kernel();
  auto q = entryPoint.qalloc();
  entryPoint.x(q);
  entryPoint.call(kernel, q);
  entryPoint.adjoint(kernel, q);
  entryPoint.mz(q);
  printf("%s\n", entryPoint.to_quake().c_str());

  auto counts = cudaq::sample(entryPoint);
  counts.dump();
  EXPECT_EQ(counts.size(), 1);
  EXPECT_EQ(counts.begin()->first, "1");
}

// Conditional execution (including reset) on the tensornet backend is slow for
// a large number of shots.
#ifndef CUDAQ_BACKEND_TENSORNET
CUDAQ_TEST(BuilderTester, checkReset) {
  {
    auto entryPoint = cudaq::make_kernel();
    auto q = entryPoint.qalloc();
    entryPoint.x(q);
    entryPoint.reset(q);
    entryPoint.mz(q);
    auto counts = cudaq::sample(entryPoint);
    EXPECT_EQ(counts.size(), 1);
    EXPECT_EQ(counts.begin()->first, "0");
  }
  {
    auto entryPoint = cudaq::make_kernel();
    auto q = entryPoint.qalloc(2);
    entryPoint.x(q);
    // For now, don't allow reset on veq.
    entryPoint.reset(q);
    entryPoint.mz(q);
    printf("%s\n", entryPoint.to_quake().c_str());

    auto counts = cudaq::sample(entryPoint);
    counts.dump();
    EXPECT_EQ(counts.size(), 1);
    EXPECT_EQ(counts.begin()->first, "00");
  }
  {
    auto entryPoint = cudaq::make_kernel();
    auto q = entryPoint.qalloc(2);
    entryPoint.x(q);
    entryPoint.reset(q[0]);
    entryPoint.mz(q);
    auto counts = cudaq::sample(entryPoint);
    EXPECT_EQ(counts.size(), 1);
    EXPECT_EQ(counts.begin()->first, "01");
  }
}
#endif

CUDAQ_TEST(BuilderTester, checkForLoop) {

  {
    auto ret = cudaq::make_kernel<std::size_t>();
    auto &circuit = ret.get<0>();
    auto &inSize = ret.get<1>();
    auto qubits = circuit.qalloc(inSize);
    circuit.h(qubits[0]);
    circuit.for_loop(0, inSize - 1, [&](auto &index) {
      circuit.x<cudaq::ctrl>(qubits[index], qubits[index + 1]);
    });

    printf("%s\n", circuit.to_quake().c_str());
    auto counts = cudaq::sample(circuit, 5);
    std::size_t counter = 0;
    for (auto &[k, v] : counts) {
      counter += v;
      EXPECT_TRUE(k == "00000" || k == "11111");
    }
    EXPECT_EQ(counter, 1000);
  }

  {
    auto ret = cudaq::make_kernel<std::size_t>();
    auto &circuit = ret.get<0>();
    auto &inSize = ret.get<1>();
    auto qubits = circuit.qalloc(inSize);
    circuit.h(qubits[0]);
    // can pass concrete integers for both
    circuit.for_loop(0, 4, [&](auto &index) {
      circuit.x<cudaq::ctrl>(qubits[index], qubits[index + 1]);
    });

    printf("%s\n", circuit.to_quake().c_str());
    auto counts = cudaq::sample(circuit, 5);
    std::size_t counter = 0;
    for (auto &[k, v] : counts) {
      counter += v;
      EXPECT_TRUE(k == "00000" || k == "11111");
    }
    EXPECT_EQ(counter, 1000);
  }

  {
    // Test that we can iterate over existing QuakeValues
    auto ret = cudaq::make_kernel<std::vector<double>>();
    auto &circuit = ret.get<0>();
    auto &params = ret.get<1>();

    // Get the size of the input params
    auto size = params.size();
    auto qubits = circuit.qalloc(size);

    // can pass concrete integers for both
    circuit.for_loop(0, size, [&](auto &index) {
      circuit.ry(params[index], qubits[index]);
    });

    printf("%s\n", circuit.to_quake().c_str());

    auto counts = cudaq::sample(circuit, std::vector<double>{1., 2.});
    counts.dump();
    // Should have 2 qubit results since this is a 2 parameter input
    EXPECT_EQ(counts.begin()->first.length(), 2);
  }

  {
    // Check for loop with a QuakeValue as the start index
    auto ret = cudaq::make_kernel<int, int>();
    auto &kernel = ret.get<0>();
    auto &start = ret.get<1>();
    auto &stop = ret.get<2>();
    auto qubits = kernel.qalloc(stop);
    kernel.h(qubits[0]);
    auto foo = [&](auto &index) { kernel.x(qubits[index]); };
    kernel.for_loop(start, 1, foo);
    kernel.for_loop(start, stop - 1, foo);
    printf("%s\n", kernel.to_quake().c_str());
    auto counts = cudaq::sample(kernel, 0, 8);
    counts.dump();
  }
}

// Conditional execution (including reset) on the tensornet backend is slow for
// a large number of shots.
#ifndef CUDAQ_BACKEND_TENSORNET
CUDAQ_TEST(BuilderTester, checkMidCircuitMeasure) {
  {
    auto entryPoint = cudaq::make_kernel();
    auto qubit = entryPoint.qalloc();
    entryPoint.x(qubit);
    entryPoint.mz(qubit, "c0");
    printf("%s\n", entryPoint.to_quake().c_str());

    auto counts = cudaq::sample(entryPoint);
    counts.dump();
    EXPECT_EQ(counts.register_names().size(), 2); // includes synthetic global
    EXPECT_EQ(counts.register_names()[0], "__global__");
    EXPECT_EQ(counts.register_names()[1], "c0");
  }

  {
    auto entryPoint = cudaq::make_kernel();
    auto qubit = entryPoint.qalloc();
    entryPoint.x(qubit);
    entryPoint.mz(qubit, "c0");
    entryPoint.x(qubit);
    entryPoint.mz(qubit, "c1");

    printf("%s\n", entryPoint.to_quake().c_str());

    auto counts = cudaq::sample(entryPoint);
    counts.dump();
    EXPECT_EQ(counts.register_names().size(), 3); // includes synthetic global
    auto regNames = counts.register_names();
    EXPECT_TRUE(std::find(regNames.begin(), regNames.end(), "c0") !=
                regNames.end());
    EXPECT_TRUE(std::find(regNames.begin(), regNames.end(), "c1") !=
                regNames.end());

    EXPECT_EQ(counts.count("0", "c1"), 1000);
    EXPECT_EQ(counts.count("1", "c0"), 1000);
  }

  {
    // Measure one qubit to one reg, and another to another reg.
    auto entryPoint = cudaq::make_kernel();
    auto q = entryPoint.qalloc(2);
    entryPoint.x(q[0]);
    entryPoint.mz(q[0], "hello");
    entryPoint.mz(q[1], "hello2");

    printf("%s\n", entryPoint.to_quake().c_str());
    auto counts = cudaq::sample(entryPoint);
    counts.dump();

    EXPECT_EQ(counts.count("1", "hello"), 1000);
    EXPECT_EQ(counts.count("0", "hello"), 0);
    EXPECT_EQ(counts.count("1", "hello2"), 0);
    EXPECT_EQ(counts.count("0", "hello2"), 1000);
  }
}
#endif

CUDAQ_TEST(BuilderTester, checkNestedKernelCall) {
  auto [kernel1, qubit1] = cudaq::make_kernel<cudaq::qubit>();
  auto [kernel2, qubit2] = cudaq::make_kernel<cudaq::qubit>();
  kernel2.call(kernel1, qubit2);
  auto kernel3 = cudaq::make_kernel();
  auto qreg3 = kernel3.qalloc(1);
  auto qubit3 = qreg3[0];
  kernel3.call(kernel2, qubit3);
  auto quake = kernel3.to_quake();
  std::cout << quake;

  auto count = [](const std::string &str, const std::string &substr) {
    std::size_t n = 0, pos = 0;
    while ((pos = str.find(substr, pos)) != std::string::npos) {
      ++n;
      pos += substr.length();
    }
    return n;
  };

  EXPECT_EQ(count(quake, "func.func"), 3);
  EXPECT_EQ(count(quake, "call @__nvqpp__"), 2);
}

CUDAQ_TEST(BuilderTester, checkEntryPointAttribute) {
  auto kernel = cudaq::make_kernel();
  auto quake = kernel.to_quake();
  std::cout << quake;

  std::regex functionDecleration(
      R"(func\.func @__nvqpp__mlirgen\w+\(\) attributes \{"cudaq-entrypoint"\})");
  EXPECT_TRUE(std::regex_search(quake, functionDecleration));
}

CUDAQ_TEST(BuilderTester, checkExpPauli) {
  std::vector<double> h2_data{
      3, 1, 1, 3, 0.0454063,  0,  2, 0, 0, 0, 0.17028,    0,
      0, 0, 2, 0, -0.220041,  -0, 1, 3, 3, 1, 0.0454063,  0,
      0, 0, 0, 0, -0.106477,  0,  0, 2, 0, 0, 0.17028,    0,
      0, 0, 0, 2, -0.220041,  -0, 3, 3, 1, 1, -0.0454063, -0,
      2, 2, 0, 0, 0.168336,   0,  2, 0, 2, 0, 0.1202,     0,
      0, 2, 0, 2, 0.1202,     0,  2, 0, 0, 2, 0.165607,   0,
      0, 2, 2, 0, 0.165607,   0,  0, 0, 2, 2, 0.174073,   0,
      1, 1, 3, 3, -0.0454063, -0, 15};
  cudaq::spin_op h(h2_data, 4);
  {
    auto [kernel, theta] = cudaq::make_kernel<double>();
    auto qubits = kernel.qalloc(4);
    kernel.x(qubits[0]);
    kernel.x(qubits[1]);
    kernel.exp_pauli(theta, qubits, "XXXY");
    std::cout << kernel << "\n";
    const double e = cudaq::observe(kernel, h, 0.11);
    EXPECT_NEAR(e, -1.13, 1e-2);
  }
  {
    auto [kernel, theta] = cudaq::make_kernel<double>();
    auto qubits = kernel.qalloc(4);
    kernel.x(qubits[0]);
    kernel.x(qubits[1]);
    kernel.exp_pauli(theta, qubits, cudaq::spin_op::from_word("XXXY"));
    std::cout << kernel << "\n";
    const double e = cudaq::observe(kernel, h, 0.11);
    EXPECT_NEAR(e, -1.13, 1e-2);
  }
  {
    auto [kernel, theta] = cudaq::make_kernel<double>();
    auto qubits = kernel.qalloc(4);
    kernel.x(qubits[0]);
    kernel.x(qubits[1]);
    kernel.exp_pauli(theta, qubits, "XXXY");
    std::cout << kernel << "\n";
    cudaq::optimizers::cobyla optimizer;
    optimizer.max_eval = 30;
    auto [e, opt] = optimizer.optimize(1, [&](std::vector<double> x) -> double {
      double e = cudaq::observe(kernel, h, x[0]);
      printf("E = %lf, %lf\n", e, x[0]);
      return e;
    });
    EXPECT_NEAR(e, -1.13, 1e-2);
  }
  {
    auto [kernel, theta] = cudaq::make_kernel<double>();
    auto qubits = kernel.qalloc(4);
    kernel.x(qubits[0]);
    kernel.x(qubits[1]);
    kernel.exp_pauli(theta, "XXXY", qubits[0], qubits[1], qubits[2], qubits[3]);
    std::cout << kernel << "\n";
    cudaq::optimizers::cobyla optimizer;
    optimizer.max_eval = 30;
    auto [e, opt] = optimizer.optimize(1, [&](std::vector<double> x) -> double {
      double e = cudaq::observe(kernel, h, x[0]);
      printf("E = %lf, %lf\n", e, x[0]);
      return e;
    });
    EXPECT_NEAR(e, -1.13, 1e-2);
  }
}

CUDAQ_TEST(BuilderTester, checkControlledRotations) {
  // rx: pi
  {
    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(2);
    auto controls2 = kernel.qalloc(2);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    // All of our controls in the 1-state.
    kernel.x(controls1);
    kernel.x(controls2);
    kernel.x(control3);

    kernel.rx<cudaq::ctrl>(M_PI, controls1, controls2, control3, target);

    std::cout << kernel.to_quake() << "\n";

    auto counts = cudaq::sample(kernel);
    counts.dump();

    // Target qubit should've been rotated to |1>.
    EXPECT_EQ(counts.count("111111"), 1000);
  }

  // rx: 0.0
  {
    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(2);
    auto controls2 = kernel.qalloc(2);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    // All of our controls in the 1-state.
    kernel.x(controls1);
    kernel.x(controls2);
    kernel.x(control3);

    kernel.rx<cudaq::ctrl>(0.0, controls1, controls2, control3, target);

    auto counts = cudaq::sample(kernel);
    counts.dump();

    // Target qubit should've stayed in |0>
    EXPECT_EQ(counts.count("111110"), 1000);
  }

  // ry: pi
  {
    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(2);
    auto controls2 = kernel.qalloc(2);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    // All of our controls in the 1-state.
    kernel.x(controls1);
    kernel.x(controls2);
    kernel.x(control3);

    kernel.ry<cudaq::ctrl>(M_PI, controls1, controls2, control3, target);

    auto counts = cudaq::sample(kernel);
    counts.dump();

    // Target qubit should've been rotated to |1>
    EXPECT_EQ(counts.count("111111"), 1000);
  }

  // ry: pi / 2
  {
    cudaq::set_random_seed(4);

    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(2);
    auto controls2 = kernel.qalloc(2);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    // All of our controls in the 1-state.
    kernel.x(controls1);
    kernel.x(controls2);
    kernel.x(control3);

    kernel.ry<cudaq::ctrl>(M_PI_2, controls1, controls2, control3, target);

    auto counts = cudaq::sample(kernel);
    counts.dump();

    // Target qubit should have a 50/50 mix between |0> and |1>
    EXPECT_TRUE(counts.count("111111") < 550);
    EXPECT_TRUE(counts.count("111110") > 450);
  }

  {
    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(3);
    auto controls2 = kernel.qalloc(3);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    kernel.x(controls1);
    kernel.x(control3);
    // Should do nothing.
    kernel.x<cudaq::ctrl>(controls1, controls2, control3, target);
    kernel.x(controls2);
    // Should rotate `target`.
    kernel.rx<cudaq::ctrl>(M_PI, controls1, controls2, control3, target);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    EXPECT_EQ(counts.count("11111111"), 1000);
  }
}

#if !defined(CUDAQ_BACKEND_DM) && !defined(CUDAQ_BACKEND_TENSORNET)

TEST(BuilderTester, checkFromStateVector) {
  std::vector<cudaq::complex> vec{M_SQRT1_2, 0., 0., M_SQRT1_2};
  {
    auto kernel = cudaq::make_kernel();
    auto qubits = kernel.qalloc(vec);
    std::cout << kernel << "\n";
    auto counts = cudaq::sample(kernel);
    counts.dump();
    EXPECT_EQ(counts.size(), 2);
    std::size_t counter = 0;
    for (auto &[k, v] : counts) {
      counter += v;
      EXPECT_TRUE(k == "00" || k == "11");
    }
    EXPECT_EQ(counter, 1000);
  }

  {
    auto [kernel, initState] =
        cudaq::make_kernel<std::vector<cudaq::complex>>();
    auto qubits = kernel.qalloc(initState);
    std::cout << kernel << "\n";
    auto counts = cudaq::sample(kernel, vec);
    counts.dump();
    EXPECT_EQ(counts.size(), 2);
    std::size_t counter = 0;
    for (auto &[k, v] : counts) {
      counter += v;
      EXPECT_TRUE(k == "00" || k == "11");
    }
    EXPECT_EQ(counter, 1000);
  }

  {
    // 2 qubit 11 state
    std::vector<cudaq::complex> vec{0., 0., 0., 1.};
    auto [kernel, initState] =
        cudaq::make_kernel<std::vector<cudaq::complex>>();
    auto qubits = kernel.qalloc(initState);
    // induce the need for a kron prod between
    // [0,0,0,1] and [1, 0, 0, 0]
    auto anotherOne = kernel.qalloc(2);
    std::cout << kernel << "\n";
    auto counts = cudaq::sample(kernel, vec);
    counts.dump();
    EXPECT_EQ(counts.size(), 1);
    EXPECT_EQ(counts.count("1100"), 1000);
  }

  {
    // 2 qubit 11 state
    std::vector<cudaq::complex> vec{0., 0., 0., 1.};
    auto [kernel, initState] =
        cudaq::make_kernel<std::vector<cudaq::complex>>();
    auto qubits = kernel.qalloc(initState);
    // induce the need for a kron prod between
    // [0,0,0,1] and [1, 0]
    auto anotherOne = kernel.qalloc();
    std::cout << kernel << "\n";
    auto counts = cudaq::sample(kernel, vec);
    counts.dump();
    EXPECT_EQ(counts.size(), 1);
    EXPECT_EQ(counts.count("110"), 1000);
  }
}

CUDAQ_TEST(BuilderTester, checkCanProgressivelyBuild) {
  auto kernel = cudaq::make_kernel();
  auto q = kernel.qalloc(2);
  kernel.h(q[0]);
  auto state = cudaq::get_state(kernel);
  EXPECT_NEAR(M_SQRT1_2, state[0].real(), 1e-3);
  // Handle sims with different endianness
  EXPECT_TRUE(std::fabs(M_SQRT1_2 - state[1].real()) < 1e-3 ||
              std::fabs(M_SQRT1_2 - state[2].real()) < 1e-3);
  EXPECT_NEAR(0.0, state[3].real(), 1e-3);

  auto counts = cudaq::sample(kernel);
  EXPECT_TRUE(counts.count("00") != 0);
  EXPECT_TRUE(counts.count("10") != 0);

  // Continue building the kernel
  kernel.x<cudaq::ctrl>(q[0], q[1]);
  state = cudaq::get_state(kernel);
  EXPECT_NEAR(M_SQRT1_2, state[0].real(), 1e-3);
  EXPECT_NEAR(0.0, state[1].real(), 1e-3);
  EXPECT_NEAR(0.0, state[2].real(), 1e-3);
  EXPECT_NEAR(M_SQRT1_2, state[3].real(), 1e-3);

  counts = cudaq::sample(kernel);
  EXPECT_TRUE(counts.count("00") != 0);
  EXPECT_TRUE(counts.count("11") != 0);
}

CUDAQ_TEST(BuilderTester, checkQuakeValueOperators) {
  // Test arith operators on QuakeValue
  auto [kernel1, theta] = cudaq::make_kernel<double>();
  auto q1 = kernel1.qalloc(1);
  kernel1.rx(theta / 8.0, q1[0]);
  auto state1 = cudaq::get_state(kernel1, M_PI);

  auto [kernel2, factor] = cudaq::make_kernel<double>();
  auto q2 = kernel2.qalloc(1);
  kernel2.rx(M_PI / factor, q2[0]);
  auto state2 = cudaq::get_state(kernel2, 8.0);

  auto [kernel3, arg1, arg2] = cudaq::make_kernel<double, double>();
  auto q3 = kernel3.qalloc(1);
  kernel3.rx(arg1 / arg2, q3[0]);
  auto state3 = cudaq::get_state(kernel3, M_PI, 8.0);

  // Reference
  auto kernel = cudaq::make_kernel();
  auto q = kernel.qalloc(1);
  kernel.rx(M_PI / 8.0, q[0]);
  auto state = cudaq::get_state(kernel);

  EXPECT_NEAR(state.overlap(state1).real(), 1.0, 1e-3);
  EXPECT_NEAR(state.overlap(state2).real(), 1.0, 1e-3);
  EXPECT_NEAR(state.overlap(state3).real(), 1.0, 1e-3);
}

#endif
