/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This is code to be included into parallel tests. The tests shall be separate
// files for llvm-lit.

#include <cudaq.h>
#include <iostream>
#include <string>
#include <vector>

__qpu__ void callee(cudaq::qview<> r) {
  for (auto i = 0; i < 4; i++) {
    if (i % 2 == 0)
      x(r[i]);

    auto m = mz(r[i]);
    cudaq::reset(r[i]);

    if (m)
      x(r[i]);
    else
      h(r[i]);
  }
}

struct caller {
  void operator()() __qpu__ {
    cudaq::qvector q(4);
    callee(q);
  }
};

__qpu__ void c_caller() {
  cudaq::qvector q(4);
  callee(q);
}

struct inlined {
  void operator()() __qpu__ {
    cudaq::qvector r(4);
    for (auto i = 0; i < 4; i++) {
      if (i % 2 == 0)
        x(r[i]);

      auto m = mz(r[i]);
      cudaq::reset(r[i]);

      if (m)
        x(r[i]);
      else
        h(r[i]);
    }
  }
};

__qpu__ void c_inlined() {
  cudaq::qvector r(4);
  for (auto i = 0; i < 4; i++) {
    if (i % 2 == 0)
      x(r[i]);

    auto m = mz(r[i]);
    cudaq::reset(r[i]);

    if (m)
      x(r[i]);
    else
      h(r[i]);
  }
}

int main() {
  {
    auto counts = cudaq::sample(1000, caller{});
    counts.dump();

    printf("%d\n", counts.count("1010", "__global__") > 100);
    printf("%d\n", counts.count("1011", "__global__") > 100);
    printf("%d\n", counts.count("1110", "__global__") > 100);
    printf("%d\n", counts.count("1111", "__global__") > 100);
  }

  {
    auto counts = cudaq::sample(1000, inlined{});
    counts.dump();

    printf("%d\n", counts.count("1010", "__global__") > 100);
    printf("%d\n", counts.count("1011", "__global__") > 100);
    printf("%d\n", counts.count("1110", "__global__") > 100);
    printf("%d\n", counts.count("1111", "__global__") > 100);
  }

  // Issue: https://github.com/NVIDIA/cuda-quantum/issues/3215
  // {
  //   auto counts = cudaq::sample(1000, c_caller);
  //   counts.dump();

  //   printf("%zu\n", counts.count("1010", "__global__") > 100);
  //   printf("%zu\n", counts.count("1011", "__global__") > 100);
  //   printf("%zu\n", counts.count("1110", "__global__") > 100);
  //   printf("%zu\n", counts.count("1111", "__global__") > 100);
  // }

  // {
  //   auto counts = cudaq::sample(1000, c_inlined);
  //   counts.dump();

  //   printf("%zu\n", counts.count("1010", "__global__") > 100);
  //   printf("%zu\n", counts.count("1011", "__global__") > 100);
  //   printf("%zu\n", counts.count("1110", "__global__") > 100);
  //   printf("%zu\n", counts.count("1111", "__global__") > 100);
  // }

  printf("%s", "done");
  return 0;
}
