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

struct Foo {
  bool b;
  cudaq::measure_result r;
};

__qpu__ auto return_struct_with_measure_result() { // expected-error{{kernel result type not supported}}
  cudaq::qubit q;
  h(q);
  Foo foo{false, mz(q)};
  return foo;
};

int main() {
  auto const result = cudaq::run(10, return_struct_with_measure_result);
  return 0;
}
