/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o %t && %t

#include "cudaq.h"
#include <cstdio>

struct VectorBoolResult {
  std::vector<bool> operator()() __qpu__ {
    std::vector<bool> result(3);
    result[0] = true;
    result[1] = false;
    result[2] = true;
    return result;
  }
};

struct VectorIntResult {
  std::vector<int> operator()() __qpu__ {
    std::vector<int> result(2);
    result[0] = 42;
    result[1] = -23479;
    return result;
  }
};

struct VectorDoubleResult {
  std::vector<double> operator()() __qpu__ {
    std::vector<double> result(2);
    result[0] = 543.0;
    result[1] = -234234.0;
    return result;
  }
};

int main() {
  auto retb{VectorBoolResult{}()};
  printf("%d %d %d\n", static_cast<int>(retb[0]), static_cast<int>(retb[1]),
         static_cast<int>(retb[2]));
  auto ret = VectorIntResult{}();
  printf("%d %d\n", ret[0], ret[1]);
  std::vector<double> retd{VectorDoubleResult{}()};
  printf("%f %f\n", retd[0], retd[1]);
  return !(retb[0] && !retb[1] && retb[2] && ret[0] == 42 && ret[1] == -23479 &&
           retd[0] == 543.0 && retd[1] == -234234.0);
}
