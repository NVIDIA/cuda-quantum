/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s -verify

#include "cudaq.h"

// expected-error@+1 {{hybrid quantum-classical struct types are not allowed}}
struct test {
  int i;
  double d;
  cudaq::qview<> q;
};

__qpu__ void hello(cudaq::qubit &q) { h(q); }

// expected-error@+1 {{failed to generate type for kernel function}}
__qpu__ void kernel(test t) {
  h(t.q);
  hello(t.q[0]);
}

__qpu__ void entry(int i) {
  cudaq::qvector q(i);
  test tt{1, 2.2, q};
  // this fails non-default ctor ConvertExpr:2899, 
  // but this is not what we are testing here
  // kernel(tt); 
}
