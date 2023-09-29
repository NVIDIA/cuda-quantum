/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %basename_t.x --target quantinuum --emulate && ./%basename_t.x | FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ void bar(cudaq::qubit& q) {
  x(q);
}

struct baz {
  __qpu__ void operator()(cudaq::qubit& q) {
    x(q);
  }
};

struct foo {
  template <typename CallableKernel>
  __qpu__ void operator()(CallableKernel &&func, int size) {
    cudaq::qreg q(size);
    func(q[0]);
    mz(q[0]);
  }
};

int main() {
  auto result = cudaq::sample(1000, foo{}, baz{}, /*qreg size*/ 1);

#ifndef SYNTAX_CHECK
  std::cout << result.most_probable() << '\n';
  assert("1" == result.most_probable());
#endif

  return 0;
}

// CHECK: 1
