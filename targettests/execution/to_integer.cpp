/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -std=c++17 --enable-mlir %s -o %t
// RUN: if [ $(echo %cpp_std | cut -c4- ) -ge 20 ]; then \
// RUN:   nvq++ --enable-mlir %s -o %t && %t; \
// RUN: fi

#include <cudaq.h>

struct test {
  int operator()(std::vector<int> applyX) __qpu__ {
    cudaq::qvector q(applyX.size());

    for (std::size_t i = 0; i < applyX.size(); i++) {
      if (applyX[i]) {
        x(q[i]);
      }
    }

    return cudaq::to_integer(mz(q));
  }
};

int main() {
  std::vector<int> firstTest{1, 1, 1}, secondTest{1, 1, 1, 1},
      thirdTest{1, 0, 1};
  auto i = test{}(firstTest);
  assert(i == 7 && "111 has to map to 7.");

  i = test{}(secondTest);
  assert(i == 15 && "1111 has to map to 15.");

  i = test{}(thirdTest);
  assert(i == 5 && "101 has to map to 15.");
}
