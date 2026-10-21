/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: not cudaq-quake %s 2>&1 | FileCheck %s

#include <cudaq.h>
#include <vector>

// The checks of the data members of a kernel class also apply to a class that
// is an instantiated template, and to a call operator that is defined outside
// of the class.

template <class T>
struct Templated {
  void operator()() __qpu__ { cudaq::qubit q; }
  // CHECK-DAG: :[[@LINE+1]]:{{[0-9]+}}: error: CUDA-Q kernel class data member must be of an arithmetic or enumeration type
  T value;
};

struct OutOfLine {
  void operator()() __qpu__;
  // CHECK-DAG: :[[@LINE+1]]:{{[0-9]+}}: error: CUDA-Q kernel class data member may not have a quantum type
  cudaq::qubit q;
};
void OutOfLine::operator()() __qpu__ {}

int main() {
  Templated<std::vector<int>> t;
  cudaq::sample(t);
  cudaq::sample(OutOfLine{});
  return 0;
}
