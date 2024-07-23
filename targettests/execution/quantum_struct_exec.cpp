/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s

#include "cudaq.h"

struct test {
  int i;
  double d;
  cudaq::qview<> q;
};

__qpu__ void hello(cudaq::qubit &q) { h(q); }

__qpu__ void kernel(test t) {
  h(t.q);
  for (int i = 0; i < t.i; i++)
    hello(t.q[i]);
}

__qpu__ void entry(int i) {
  cudaq::qvector q(i);
  test tt{i, 2.2, q};
  kernel(tt);
}

int main() {
  auto counts = cudaq::sample(entry, 4);
  assert(counts.size() == 1);
  assert(counts.begin()->first == "0000");
}