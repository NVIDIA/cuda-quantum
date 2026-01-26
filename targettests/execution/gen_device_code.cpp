/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s --enable-mlir -fno-aggressive-inline -o %t && %t | \
// RUN: FileCheck %s

#include <cudaq.h>
#include <iostream>

struct bax {
  void operator()(cudaq::qubit &q) __qpu__ { x(q); }
};

struct baz {
  void operator()(cudaq::qubit &q) __qpu__ { z(q); }
};

struct bah {
  void operator()(cudaq::qubit &q) __qpu__ { h(q); }
};

struct bar1 {
  void operator()(cudaq::qubit &q) __qpu__ {
    bax{}(q);
    baz{}(q);
  }
};

struct bar2 {
  void operator()(cudaq::qubit &q) __qpu__ {
    bah{}(q);
    bax{}(q);
    bar1{}(q);
  }
};

struct foo {
  void operator()() __qpu__ {
    cudaq::qubit q;
    bar2{}(q);
    bax{}(q);
    mz(q);
  }
};

int main() {
  std::cout << cudaq::get_quake(foo{}) << "\n";
  return 0;
}
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__foo
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__bar2
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__bah
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__bax
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__bar1
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__baz
