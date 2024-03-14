/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --emit-qir %s && cat struct_arg.qir.ll | \
// RUN: FileCheck %s ; rm struct_arg.qir.ll

#include <cudaq.h>
#include <iostream>

struct baz {
  void operator()(cudaq::qubit &q) __qpu__ { x(q); }
};

struct foo {
  template <typename CallableKernel>
  void operator()(CallableKernel &&func, int n) __qpu__ {
    cudaq::qvector q(n);
    func(q[0]);
    mz(q[0]);
  }
};

// clang-format off
// CHECK-LABEL: define void @_ZN3fooclI3bazEEvOT_i
// CHECK-SAME: (i8* nocapture readnone %{{.*}}, {}* nocapture readnone %{{.*}}, i32 %{{.*}})
// clang-format on

int main() {
  auto result = cudaq::sample(1000, foo{}, baz{}, 1);
  for (auto &&[bits, counts] : result) {
    std::cout << bits << " : " << counts << '\n';
  }
  return 0;
}
