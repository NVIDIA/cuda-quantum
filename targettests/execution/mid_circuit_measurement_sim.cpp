/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Simulators
// RUN: nvq++ --enable-mlir  %s -o %t && %t | FileCheck %s
// RUN: nvq++ --library-mode %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iostream>
#include <string>
#include <vector>

__qpu__ std::vector<int> callee(cudaq::qview<> r) {
  std::vector<int> result = {0, 0, 0, 0, 0, 0, 0, 0};
  for (auto i = 0; i < 4; i++) {
    auto j = i * 2;
    if (i % 2 == 0)
      x(r[i]);

    result[j] = mz(r[i]);
    cudaq::reset(r[i]);

    if (result[j])
      x(r[i]);
    else
      h(r[i]);

    result[j + 1] = mz(r[i]);
  }
  return result;
}

struct caller {
  std::vector<int> operator()() __qpu__ {
    cudaq::qvector q(4);
    return callee(q);
  }
};

__qpu__ std::vector<int> c_caller() {
  cudaq::qvector q(4);
  return callee(q);
}

struct inlined {
  std::vector<int> operator()() __qpu__ {
    std::vector<int> result = {0, 0, 0, 0, 0, 0, 0, 0};
    cudaq::qvector r(4);
    for (auto i = 0; i < 4; i++) {
      auto j = i * 2;
      if (i % 2 == 0)
        x(r[i]);

      result[j] = mz(r[i]);
      cudaq::reset(r[i]);

      if (result[j])
        x(r[i]);
      else
        h(r[i]);

      result[j + 1] = mz(r[i]);
    }
    return result;
  }
};

__qpu__ std::vector<int> c_inlined() {
  std::vector<int> result = {0, 0, 0, 0, 0, 0, 0, 0};
  cudaq::qvector r(4);
  for (auto i = 0; i < 4; i++) {
    auto j = i * 2;
    if (i % 2 == 0)
      x(r[i]);

    result[j] = mz(r[i]);
    cudaq::reset(r[i]);

    if (result[j])
      x(r[i]);
    else
      h(r[i]);

    result[j + 1] = mz(r[i]);
  }
  return result;
}

void validate_results(const std::vector<std::vector<int>> &results) {
  assert(results.size() == 10);
  for (auto &result : results) {
    assert(result.size() == 8);
    assert(result[0] == 1 && result[1] == 1); // qubit 0
    assert(result[2] == 0);                   // qubit 1
    assert(result[3] == 0 || result[3] == 1); // qubit 1
    assert(result[4] == 1 && result[5] == 1); // qubit 2
    assert(result[6] == 0);                   // qubit 3
    assert(result[7] == 0 || result[7] == 1); // qubit 3
  }
  printf("%s", "1\n");
}

int main() {
  validate_results(cudaq::run(10, caller{}));
  validate_results(cudaq::run(10, inlined{}));
  validate_results(cudaq::run(10, c_caller));
  validate_results(cudaq::run(10, c_inlined));
  printf("%s", "done\n");
  return 0;
}

// CHECK: 1
// CHECK: 1
// CHECK: 1
// CHECK: 1
// CHECK: done
