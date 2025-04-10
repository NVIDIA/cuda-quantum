/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target infleqtion      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target ionq                     --emulate %s -o %t && %t | FileCheck %s
// 2 different IQM machines for 2 different topologies
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Adonis --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Apollo --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target oqc                      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target quantinuum               --emulate %s -o %t && %t | FileCheck %s
// RUN: if %braket_avail; then nvq++ %cpp_std --target braket --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <cudaq.h>
#include <iostream>
#include <string>
#include <vector>

struct testConstBool {
  auto operator()() __qpu__ {
    cudaq::qvector q(4);
    unsigned i0 = (unsigned)(true);
    if (i0 == 1) {
      x(q[0]);
    }
    unsigned i1 = (unsigned)(false);
    if (i1 == 0) {
      x(q[1]);
    }
    signed i2 = (signed)(true);
    if (i2 == 1) {
      x(q[2]);
    }
    signed i3 = (signed)(false);
    if (i3 == 0) {
      x(q[3]);
    }
  }
};

struct testBool {
  auto operator()() __qpu__ {
    cudaq::qvector q(4);
    bool t = mz(q[0]);
    bool b = !t;
    unsigned i0 = (unsigned)(b);
    if (i0 == 1) {
      x(q[0]);
    }
    unsigned i1 = (unsigned)(!b);
    if (i1 == 0) {
      x(q[1]);
    }
    signed i2 = (signed)(b);
    if (i2 == 1) {
      x(q[2]);
    }
    signed i3 = (signed)(!b);
    if (i3 == 0) {
      x(q[3]);
    }
  }
};

struct testInt {
  auto operator()(int n) __qpu__ {
    cudaq::qvector q(2);
    unsigned i0 = (unsigned)(n);
    if (i0) {
      x(q[0]);
    }
    signed i2 = (signed)(n);
    if (i2) {
      x(q[1]);
    }
  }
};


void printCounts(cudaq::sample_result &result) {
  std::vector<std::string> values{};
  for (auto &&[bits, counts] : result) {
    values.push_back(bits);
  }

  std::sort(values.begin(), values.end());
  for (auto &&bits : values) {
    std::cout << bits << std::endl;
  }
}

int main() {
  {
    std::string expected = "";
    unsigned i0 = (unsigned)(true);
    expected += i0 == 1 ? "1" : "0";
    unsigned i1 = (unsigned)(false);
    expected += i1 == 0 ? "1" : "0";
    signed i2 = (signed)(true);
    expected += i2 == 1 ? "1" : "0";
    signed i3 = (signed)(false);
    expected += i3 == 0 ? "1" : "0";

    printf("%s\n", expected.c_str());

    auto counts = cudaq::sample(testConstBool{});
    printCounts(counts);

    counts = cudaq::sample(testBool{});
    printCounts(counts);
  }

  {
    std::string expected = "";
    unsigned i0 = (unsigned)(-1);
    expected += i0 ? "1" : "0";
    signed i2 = (signed)(-1);
    expected += i2? "1" : "0";

    printf("%s\n", expected.c_str());

    auto counts = cudaq::sample(testInt{}, -1);
    printCounts(counts);
  }
  return 0;
}

// CHECK: 1111
// CHECK: 1111
// CHECK: 1111
