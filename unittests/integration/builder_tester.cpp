/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <cudaq/algorithms/gradients/central_difference.h>
#include <cudaq/builder.h>
#include <cudaq/optimizers.h>

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
    // invalidates the qreg reference you have.
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
    auto [opt_val_0, optpp] =
        cudaq::vqe(ansatz, gradient, h3, optimizer, 2, argMapper);
    printf("HELLO %lf %lf \n", optpp[0], optpp[1]);
    printf("<H3> = %lf\n", opt_val_0);
    EXPECT_NEAR(opt_val_0, -2.045375, 1e-3);
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
    auto [opt_val_0, optpp] = cudaq::vqe(ansatz, gradient, h3, optimizer, 2);
    printf("<H3> = %lf\n", opt_val_0);
    EXPECT_NEAR(opt_val_0, -2.045375, 1e-3);
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
  }
}

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

CUDAQ_TEST(BuilderTester, checkQregArg) {
  auto [kernel, qregArg] = cudaq::make_kernel<cudaq::qreg<>>();
  kernel.h(qregArg);

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

  // Must provide the number of parameters that were extracted
  EXPECT_ANY_THROW({
    auto counts =
        cudaq::sample(kernel, std::vector<double>{M_PI, M_PI_2, M_PI});
  });
}

CUDAQ_TEST(BuilderTester, checkIsArgStdVec) {
  auto [kernel, one, two, thetas, four] =
      cudaq::make_kernel<double, float, std::vector<double>, int>();

  EXPECT_TRUE(kernel.isArgStdVec(2));
  EXPECT_FALSE(kernel.isArgStdVec(1));
}

CUDAQ_TEST(BuilderTester, checkKernelControl) {

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
  auto counts = cudaq::sample(hadamardTest);
  counts.dump();
  printf("< 1 | X | 1 > = %lf\n", counts.exp_val_z());
  EXPECT_NEAR(counts.exp_val_z(), 0.0, 1e-1);

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
  counts = cudaq::sample(hadamardTest2);
  printf("< 1 | H | 1 > = %lf\n", counts.exp_val_z());
  EXPECT_NEAR(counts.exp_val_z(), -1.0 / std::sqrt(2.0), 1e-1);

  // Demonstrate can control on qreg
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

CUDAQ_TEST(BuilderTester, checkReset) {
  {
    auto entryPoint = cudaq::make_kernel();
    auto q = entryPoint.qalloc();
    entryPoint.x(q);
    entryPoint.reset(q);
    entryPoint.mz(q);
    printf("%s\n", entryPoint.to_quake().c_str());

    auto counts = cudaq::sample(entryPoint);
    counts.dump();
    EXPECT_EQ(counts.size(), 1);
    EXPECT_EQ(counts.begin()->first, "0");
  }
  {
    auto entryPoint = cudaq::make_kernel();
    auto q = entryPoint.qalloc(2);
    entryPoint.x(q);
    // For now, don't allow reset on qvec.
    entryPoint.reset(q);
    entryPoint.mz(q);
    printf("%s\n", entryPoint.to_quake().c_str());

    auto counts = cudaq::sample(entryPoint);
    counts.dump();
    EXPECT_EQ(counts.size(), 1);
    EXPECT_EQ(counts.begin()->first, "00");
  }
}

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
}