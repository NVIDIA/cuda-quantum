/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>
#include <gtest/gtest.h>

TEST(HostDeviceMatrixExpValTester, BasicCheck) {
  EXPECT_TRUE(std::getenv("CUDAQ_MAX_CPU_MEMORY_GB") != nullptr);
  const auto cpuMemGb = std::atoi(std::getenv("CUDAQ_MAX_CPU_MEMORY_GB"));
  // Some very large value
  EXPECT_GE(cpuMemGb, 1024);
}

TEST(HostDeviceMatrixExpValTester, checkSimple) {
  constexpr int numQubits = 33; // Large number of qubits
  auto ansatz = [&]() __qpu__ {
    cudaq::qvector q(numQubits);
    x(q);
  };

  auto ham = cudaq::spin_op::z(0);
  auto result = cudaq::observe(ansatz, ham);
  std::cout << "Exp val = " << result.expectation() << "\n";
  EXPECT_NEAR(result.expectation(), -1.0, 1e-6);
}

TEST(HostDeviceMatrixExpValTester, checkResult) {
  constexpr int numQubits = 33; // Large number of qubits
  // Create a special hamiltonian with the first and last qubits
  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(numQubits - 1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(numQubits - 1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(numQubits - 1);
  auto ansatz = [&](double theta) __qpu__ {
    cudaq::qvector q(numQubits);
    x(q[0]);
    ry(theta, q[numQubits - 1]);
    x<cudaq::ctrl>(q[numQubits - 1], q[0]);
  };

  double result = cudaq::observe(ansatz, h, 0.59);
  printf("Energy value = %lf\n", result);
  EXPECT_NEAR(result, -1.7487, 1e-3);
}

TEST(HostDeviceMatrixExpValTester, checkWidePauliTerm) {
  constexpr int numOps = 20;
  auto ham = cudaq::spin_op::z(0);
  for (int i = 1; i < numOps; ++i)
    ham *= cudaq::spin_op::z(i);

  auto referenceAnsatz = [&]() __qpu__ {
    cudaq::qvector q(numOps);
    x(q);
  };
  const double reference = cudaq::observe(referenceAnsatz, ham);

  constexpr int numQubits = 33;
  auto migratedAnsatz = [&]() __qpu__ {
    cudaq::qvector q(numQubits);
    x(q);
  };
  const double migrated = cudaq::observe(migratedAnsatz, ham);
  EXPECT_NEAR(migrated, reference, 1e-6);
}
