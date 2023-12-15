/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++                               %s -o %t && %t | FileCheck %s
// RUN: nvq++ -std=c++17 %s --enable-mlir -o %t
// clang-format on

#include <cudaq.h>
#include <iostream>

int main() {
  const int nShots = 1000;

  // Check that qubits show up in measurement order (w/o var names)
  auto test1 = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
    mz(b);
    mz(a);
  };
  auto result = cudaq::sample(nShots, test1);
  auto globalRegResults = cudaq::sample_result{
      cudaq::ExecutionResult{result.to_map(cudaq::GlobalRegisterName)}};
  std::cout << "test1:\n";
  globalRegResults.dump();
  // CHECK: test1:
  // CHECK: { 01:1000 }

  // Check that qubits show up in measurement order (w/ var names)
  auto test2 = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
    auto ret_b = mz(b);
    auto ret_a = mz(a);
  };
  result = cudaq::sample(nShots, test2);
  globalRegResults = cudaq::sample_result{
      cudaq::ExecutionResult{result.to_map(cudaq::GlobalRegisterName)}};
  std::cout << "test2:\n";
  globalRegResults.dump();
  // CHECK: test2:
  // CHECK: { 01:1000 }

  // Check that duplicate measurements don't get duplicated in global bistring
  auto test3 = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
    auto ma1 = mz(a); // 1st measurement of qubit a
    auto ma2 = mz(a); // 2nd measurement of qubit a
    auto mb = mz(b);
  };
  result = cudaq::sample(nShots, test3);
  globalRegResults = cudaq::sample_result{
      cudaq::ExecutionResult{result.to_map(cudaq::GlobalRegisterName)}};
  std::cout << "test3:\n";
  globalRegResults.dump();
  // CHECK: test3:
  // CHECK: { 10:1000 }

  // Check that measurements will be implicitly added to kernels that have no
  // measurements.
  auto test4 = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
  };
  result = cudaq::sample(nShots, test4);
  globalRegResults = cudaq::sample_result{
      cudaq::ExecutionResult{result.to_map(cudaq::GlobalRegisterName)}};
  std::cout << "test4:\n";
  globalRegResults.dump();
  // CHECK: test4:
  // CHECK: { 10:1000 }

  // Check that specifying a measurement on `a` hides `b` from the global
  // register.
  auto test5 = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
    mz(b);
  };
  result = cudaq::sample(nShots, test5);
  globalRegResults = cudaq::sample_result{
      cudaq::ExecutionResult{result.to_map(cudaq::GlobalRegisterName)}};
  std::cout << "test5:\n";
  globalRegResults.dump();
  // CHECK: test5:
  // CHECK: { 0:1000 }

  // Check that performing a quantum operation after the final measurement makes
  // all qubits appear in the global register.
  auto test6 = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
    mz(b);
    x(b); // note that this is not allowed in base profile programs
  };
  result = cudaq::sample(nShots, test6);
  globalRegResults = cudaq::sample_result{
      cudaq::ExecutionResult{result.to_map(cudaq::GlobalRegisterName)}};
  std::cout << "test6:\n";
  globalRegResults.dump();
  // CHECK: test6:
  // CHECK: { 11:1000 }

  return 0;
}