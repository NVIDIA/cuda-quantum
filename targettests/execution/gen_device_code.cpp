/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std %s --enable-mlir --no-aggressive-early-inline -o %t && %t | \
// RUN: FileCheck %s

#include <cudaq.h>
#include <iostream>

struct bax {
  void operator()(cudaq::qubit &q) __qpu__ { x(q); }
};

struct baz {
  void operator()(cudaq::qubit &q) __qpu__ { z(q); }
};

struct bar {
  void operator()(cudaq::qubit &q) __qpu__ {
    bax{}(q);
    baz{}(q);
  }
};

struct foo {
  void operator()() __qpu__ {
    cudaq::qubit q;
    bar{}(q);
    mz(q);
  }
};

int main() {
  cudaq::sample(foo{});
  std::cout << cudaq::get_quake(foo{}) << "\n";
  return 0;
}
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__foo
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__bar
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__bax
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__baz
