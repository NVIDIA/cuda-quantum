/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// Simulators
// RUN: nvq++ --enable-mlir  %s -o %t && %t | FileCheck %s
// RUN: nvq++ --library-mode %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>
#include <string>
#include <vector>

struct kernel_vbool {
  void operator()(int n, std::vector<int> &vec) __qpu__ {
    cudaq::qvector q(n);
    for (std::size_t i = 0; i < vec.size(); i++) {
      x(q[i]);
    }
  }
};

struct kernel_vvbool {
  void operator()(int n, std::vector<std::vector<int>> &vec) __qpu__ {
    cudaq::qvector q(n);
    for (std::size_t i = 0; i < vec.size(); i++) {
      auto &inner = vec[i];
      for (std::size_t j = 0; j < inner.size(); j++) {
        auto m = i * inner.size() + j;
        x(q[m]);
      }
    }
  }
};

struct kernel_vvvbool {
  void operator()(int n,
                  std::vector<std::vector<std::vector<int>>> &vec) __qpu__ {
    cudaq::qvector q(n);
    for (std::size_t i = 0; i < vec.size(); i++) {
      auto &inner = vec[i];
      for (std::size_t j = 0; j < inner.size(); j++) {
        auto &inner2 = inner[j];
        for (std::size_t k = 0; k < inner2.size(); k++) {
          auto m = i * inner.size() * inner2.size() + inner2.size() * j + k;
          x(q[m]);
        }
      }
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
  std::vector<int> vbool1{0, 1, 2, 3};
  std::vector<int> vbool2{4, 5, 6, 7};
  std::vector<int> vbool3{8, 9, 10, 11};
  std::vector<int> vbool4{12, 13, 14, 15};
  std::vector<std::vector<int>> vvbool1{vbool1, vbool2};
  std::vector<std::vector<int>> vvbool2{vbool3, vbool4};
  std::vector<std::vector<std::vector<int>>> vvvbool{vvbool1, vvbool2};

  auto counts = cudaq::sample(kernel_vbool{}, 4, vbool1);
  printCounts(counts);

  counts = cudaq::sample(kernel_vvbool{}, 8, vvbool1);
  printCounts(counts);

  counts = cudaq::sample(kernel_vvvbool{}, 16, vvvbool);
  printCounts(counts);
  return 0;
}

// CHECK: 1111
// CHECK: 11111111
// CHECK: 1111111111111111
