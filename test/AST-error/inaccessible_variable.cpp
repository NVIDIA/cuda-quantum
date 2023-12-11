/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
    cudaq::qvector q(i);
    // expected-error@-1 {{symbol is not accessible in this kernel}}
    // expected-error@-2 {{statement not supported in qpu kernel}}
    mz(q);
    // expected-error@-1 {{symbol is not accessible in this kernel}}
    // expected-error@-2 {{statement not supported in qpu kernel}}
  };
  Kernel{}(f);
  return 0;
}
