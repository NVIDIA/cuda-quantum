/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// RUN: nvq++-pp %s --remove-comments | FileCheck %s

__qpu__ void test0() {
  cudaq::qubit q;
  x(q);
  auto a = mz(q);
  // CHECK: auto a = mz(q, "a");
}

__qpu__ void test1() {
  cudaq::qvector q(3);
  x(q);
  auto a = mz(q);
  // CHECK: auto a = mz(q, "a");
}

__qpu__ void test2() {
  cudaq::qubit q, r;
  h(q);
  auto a = mz(q);
  // CHECK: auto a = mz(q, "a");
  if (a)
    x(r);
}

__qpu__ void test3(const int n_iter) {
  cudaq::qubit q0;
  for (int i = 0; i < n_iter; i++) {
    h(q0);
    auto q0result = mz(q0);
    // CHECK: auto q0result = mz(q0, "q0result");
    if (q0result)
      break; 
  }
}

__qpu__ void do_nothing_test() {
    cudaq::qubit q;
    mz(q);
    // CHECK: mz(q);
}

__qpu__ void test5(const int n_iter) {
  cudaq::qubit q0;
  cudaq::qubit q1;
  for (int i = 0; i < n_iter; i++) {
    h(q0);
    if (mz(q0))
      x(q1); 
  }
  auto q1result = mz(q1); 
  // CHECK: auto q1result = mz(q1, "q1result");
}

__qpu__ void test6(const int n_iter) {
  cudaq::qubit q0;
  cudaq::qubit q1;
  for (int i = 0; i < n_iter; i++) {
    h(q0);
    auto q0result = mz(q0);
    // CHECK: auto q0result = mz(q0, "q0result");
    if (q0result)
      x(q1); 
  }
  auto q1result = mz(q1);
  // CHECK: auto q1result = mz(q1, "q1result");
}

__qpu__ void test7(const int n_iter) {
  cudaq::qubit q0;
  cudaq::qubit q1;
  std::vector<cudaq::measure_result> resultVector(n_iter);
  for (int i = 0; i < n_iter; i++) {
    h(q0);
    resultVector[i] = mz(q0);
    // CHECK: resultVector[i] = mz(q0, "resultVector%" + std::to_string(i));
    if (resultVector[i])
      x(q1);
  }
  auto q1result = mz(q1);
  // CHECK: q1result = mz(q1, "q1result");
}

struct ghz {
  auto operator()(int N) __qpu__ {
    cudaq::qvector q(N);
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    auto reg = mz(q);
    // CHECK: auto reg = mz(q, "reg");
  }
};

void testLambda() {
  auto lam = []() __qpu__ {
    cudaq::qubit q; 
    h(q);
    auto reg = mz(q);
    // CHECK: auto reg = mz(q, "reg");
  };
}