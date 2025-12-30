/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t
// BROKEN: nvq++ -fkernel-exec-kind=2 %s -o %t && %t

#include <cstdio>
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
  {
    std::vector<int> firstTest{1, 1, 1}, secondTest{1, 1, 1, 1},
        thirdTest{1, 0, 1}, fourthTest{1, 0, 0, 0}, fifthTest{0, 0, 0, 1};
    auto i = test{}(firstTest);
    if (i != 7) {
      printf("111 has to map to 7.\n");
      return 1;
    }

    i = test{}(secondTest);
    if (i != 15) {
      printf("1111 has to map to 15.\n");
      return 1;
    }

    i = test{}(thirdTest);
    if (i != 5) {
      printf("101 has to map to 5.\n");
      return 1;
    }

    i = test{}(fourthTest);
    if (i != 1) {
      printf("1000 has to map to 1.\n");
      return 1;
    }

    i = test{}(fifthTest);
    if (i != 8) {
      printf("0001 has to map to 8.\n");
      return 1;
    }
  }

  // Compare to cudaq::to_integer() call on the host
  {
    std::vector<bool> firstTest{true, true, true},
        secondTest{true, true, true, true}, thirdTest{true, false, true},
        fourthTest{true, false, false, false},
        fifthTest{false, false, false, true};
    auto i = cudaq::to_integer(firstTest);
    if (i != 7) {
      printf("Host: 111 has to map to 7.\n");
      return 1;
    }

    i = cudaq::to_integer(secondTest);
    if (i != 15) {
      printf("Host: 1111 has to map to 15.\n");
      return 1;
    }

    i = cudaq::to_integer(thirdTest);
    if (i != 5) {
      printf("Host: 101 has to map to 5.\n");
      return 1;
    }

    i = cudaq::to_integer(fourthTest);
    if (i != 1) {
      printf("host: 1000 has to map to 1.\n");
      return 1;
    }

    i = cudaq::to_integer(fifthTest);
    if (i != 8) {
      printf("Host: 0001 has to map to 8.\n");
      return 1;
    }
  }
  return 0;
}
