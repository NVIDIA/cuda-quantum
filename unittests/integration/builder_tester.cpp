/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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

CUDAQ_TEST(BuilderTester, checkSwap) {
  cudaq::set_random_seed(13);

  // Simple two-qubit swap.
  {
    auto kernel = cudaq::make_kernel();
    auto q = kernel.qalloc(2);
    // 0th qubit into the 1-state.
    kernel.x(q[0]);
    // Swap their states and measure.
    kernel.swap(q[0], q[1]);
    // Measure.
    kernel.mz(q);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    EXPECT_NEAR(counts.count("01"), 1000, 0);
  }

  // Simple two-qubit swap.
  {
    auto kernel = cudaq::make_kernel();
    auto q = kernel.qalloc(2);
    // 1st qubit into the 1-state.
    kernel.x(q[1]);
    // Swap their states and measure.
    kernel.swap(q[0], q[1]);
    // Measure.
    kernel.mz(q);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    EXPECT_NEAR(counts.count("10"), 1000, 0);
  }
}

CUDAQ_TEST(BuilderTester, checkConditional) {
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
  counts = cudaq::sample(10000, hadamardTest2);
  printf("< 1 | H | 1 > = %lf\n", counts.exp_val_z());
  EXPECT_NEAR(counts.exp_val_z(), -1.0 / std::sqrt(2.0), 1e-1);

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
  {
    auto [kernel, theta] = cudaq::make_kernel<double>();
    auto qubits = kernel.qalloc(4);
    kernel.x(qubits[0]);
    kernel.x(qubits[1]);
    kernel.exp_pauli(theta, qubits, "XXXY");
    std::cout << kernel << "\n";
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

#ifndef CUDAQ_BACKEND_DM

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

#endif
