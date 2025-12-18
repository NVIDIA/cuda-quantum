/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test for std::vector support

// RUN: nvq++ %s -o %t && %t | FileCheck %s

#include <cstdio>
#include <cudaq.h>

struct vector3 {
  bool operator()(std::vector<double> theta) __qpu__ {
    return theta.size() == 3;
  }
};

struct vector5 {
  bool operator()(std::vector<double> theta) __qpu__ {
    return theta.size() == 5;
  }
};

struct vectorPow2a {
  bool operator()(std::vector<double> init) __qpu__ {
    cudaq::qvector q(init);
    return true;
  }
};

struct vectorPow2b {
  bool operator()(std::vector<double> init) __qpu__ {
    cudaq::qvector q(init);
    return true;
  }
};

int main() {
  bool b = true;
  std::vector<double> v = {1.0, 2.0, 3.0};
  b &= vector3{}(v);
  v.push_back(4.0);
  v.push_back(5.0);
  b &= vector5{}(v);
  v.pop_back();
  b &= vectorPow2a{}(v);
  std::vector<double> w{v};
  v.insert(v.end(), w.begin(), w.end());
  w.clear();
  b &= vectorPow2b{}(v);
  if (b)
    printf("success\n");
  return b ? 0 : ~0;
}

// CHECK: success
