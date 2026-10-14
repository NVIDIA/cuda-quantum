/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

// Initializing a struct from a vector value copy-constructs the vector, which
// the bridge does not support. This must be reported as an error. It must not
// continue lowering with a value that the initializer list does not expect,
// which used to fail an assertion.

#include <cudaq.h>

struct Foo {
  int bar;
  std::vector<int> baz;
};

struct Quark {
  Foo operator()(std::vector<int> v) __qpu__ {
    // expected-error@+2 {{C++ constructor (non-default) is not yet supported}}
    // expected-error@+1 {{statement not supported in qpu kernel}}
    return {747, v};
  }
};

int main() {
  Quark{}({1, 2});
  return 0;
}
