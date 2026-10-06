/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s -o /dev/null

#include <cudaq.h>

struct Kernel {
  template <typename A>
  void operator()(A &&a) __qpu__ {
    a();
  }
};

int main() {
  int i;
  auto f = [&]() __qpu__ { // expected-remark{{An inaccessible symbol}}
    // `i` is a host variable, cannot be used in this kernel.
    // expected-error@+2 {{statement not supported in qpu kernel}}
    // expected-error@+1 {{symbol is not accessible in this kernel}}
    cudaq::qvector q(i);
    // declaration of `q` failed, so it's not available either.
    // expected-error@+2 {{symbol is not accessible in this kernel}}
    // expected-error@+1 {{statement not supported in qpu kernel}}
    mz(q);
  };
  Kernel{}(f);
  return 0;
}
