/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --enable-mlir %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>

struct PureQuantumStruct {
  cudaq::qview<> view1;
  cudaq::qview<> view2;
};

struct Fehu {
  void operator()(cudaq::qview<> v) __qpu__ { h(v); }
};

struct Ansuz {
  void operator()(cudaq::qview<> v) __qpu__ { x(v); }
};

struct Uruz {
  void operator()(PureQuantumStruct group) __qpu__ {
    Ansuz{}(group.view1);
    Fehu{}(group.view1);
    Fehu{}(group.view2);
    Ansuz{}(group.view2);
  }
};

struct Thurisaz {
  void operator()() __qpu__ {
    cudaq::qvector v1(2);
    cudaq::qvector v2(3);
    PureQuantumStruct pqs{v1, v2};
    Uruz{}(pqs);
    mz(v1);
    mz(v2);
  }
};

int main() {
  auto result = cudaq::sample(Thurisaz{});
  int flags[1 << 5] = {0};
  for (auto &&[b, c] : result) {
    int off = std::stoi(b, nullptr, 2);
    if (off >= (1 << 5) || off < 0) {
      std::cout << "Amazingly incorrect: " << b << '\n';
      return 1;
    }
    flags[off] = 1 + c;
  }
  for (int i = 0; i < (1 << 5); ++i) {
    if (flags[i] == 0) {
      std::cout << "FAILED!\n";
      return 1;
    }
  }
  std::cout << "Wahoo!\n";
  return 0;
}

// CHECK: Wahoo
