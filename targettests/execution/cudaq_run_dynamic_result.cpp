/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t 2>&1 | FileCheck %s --check-prefix=QTM

#include <cudaq.h>

// Size determined by input argument — argument synthesis substitutes the
// concrete value, making the qubit count and return vector size statically
// known at JIT time.  Succeeds on all targets.
__qpu__ std::vector<bool> arg_size_bool(int n) {
  cudaq::qvector qs(n);
  x(qs);
  return mz(qs);
}

__qpu__ std::vector<int> arg_size_int(int n) {
  cudaq::qvector qs(n);
  x(qs);
  auto bits = mz(qs);
  std::vector<int> result(n);
  for (int i = 0; i < n; i++)
    result[i] = bits[i] ? 1 : 0;
  return result;
}

__qpu__ std::vector<float> arg_size_float(int n) {
  cudaq::qvector qs(n);
  x(qs);
  auto bits = mz(qs);
  std::vector<float> result(n);
  for (int i = 0; i < n; i++)
    result[i] = bits[i] ? 1.0f : 0.0f;
  return result;
}

// Size determined by measurement outcome — cannot be synthesized.
// Fails on hardware targets (qir-adaptive requires static qubit counts).
__qpu__ std::vector<bool> branch_vec_test(bool flip) {
  cudaq::qubit ctrl;
  if (flip)
    x(ctrl);
  bool b = mz(ctrl);
  int sz = b ? 2 : 4;
  cudaq::qvector data(sz);
  return mz(data);
}

int main() {
  auto res1 = cudaq::run(1, arg_size_bool, 3);
  printf("Bool arg:");
  for (bool b : res1[0]) printf(" %d", b ? 1 : 0);
  printf("\n");

  auto res2 = cudaq::run(1, arg_size_int, 3);
  printf("Int arg:");
  for (int v : res2[0]) printf(" %d", v);
  printf("\n");

  auto res3 = cudaq::run(1, arg_size_float, 3);
  printf("Float arg:");
  for (float v : res3[0]) printf(" %.0f", v);
  printf("\n");

  // Flush stdout before the expected Quantinuum failure so QTM checks see
  // the successful outputs before the error message.
  fflush(stdout);

  auto res4 = cudaq::run(1, branch_vec_test, false);
  auto res5 = cudaq::run(1, branch_vec_test, true);
  printf("Bool branch: %ld %ld\n", res4[0].size(), res5[0].size());
}

// CHECK: Bool arg: 1 1 1
// CHECK: Int arg: 1 1 1
// CHECK: Float arg: 1 1 1
// CHECK: Bool branch: 4 2

// QTM: Bool arg: 1 1 1
// QTM: Int arg: 1 1 1
// QTM: Float arg: 1 1 1
// QTM: error: 'quake.alloca' op must have a constant size
