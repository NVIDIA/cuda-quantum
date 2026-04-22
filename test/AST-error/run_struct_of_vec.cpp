/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>
#include <iostream>

__qpu__ std::vector<std::vector<int>> vec_of_vec() { 
  return {{1, 2}, {3, 4}}; 
}

struct Foo {
  int bar;
  std::vector<bool> baz;
};

struct Quark {
  Foo operator()() __qpu__ { // expected-error{{kernel result type not supported}}
    cudaq::qvector q(3);
    return {747, mz(q)};
  }
};

int main() {
  auto const result1 = cudaq::run(10, vec_of_vec); 
  auto const result2 = cudaq::run(10, Quark{});
  return 0;
}
