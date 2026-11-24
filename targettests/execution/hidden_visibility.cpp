/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -fvisibility=hidden %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ void flip(cudaq::qubit &q) { x(q); }

__qpu__ bool entry() {
  cudaq::qubit q;
  flip(q);
  return mz(q);
}

__attribute__((visibility("default"))) int main() {
  std::cout << "result: " << entry() << '\n';
  return 0;
}

// CHECK: result: 1
