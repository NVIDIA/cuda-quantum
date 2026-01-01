/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: if [ command -v split-file ]; then \
// RUN: split-file %s %t && \
// RUN: nvq++ --enable-mlir -c %t/udedgulib.cpp -o %t/udedgulib.o && \
// RUN: nvq++ --enable-mlir -c %t/udedguuser.cpp -o %t/udedguuser.o && \
// RUN: nvq++ --enable-mlir %t/udedgulib.o %t/udedguuser.o -o %t/udedgu.x && \
// RUN: %t/udedgu.x | FileCheck %s ; else \
// RUN: echo "skipping" ; fi
// clang-format on

//--- udedgulib.h

#include "cudaq.h"

__qpu__ void dunkadee(cudaq::qvector<> &q);

//--- udedgulib.cpp

#include "udedgulib.h"

__qpu__ void dunkadee(cudaq::qvector<> &q) { x(q[0]); }

//--- udedguuser.cpp

#include "udedgulib.h"
#include <iostream>

__qpu__ void userKernel(const cudaq::qkernel<void(cudaq::qvector<> &)> &init) {
  cudaq::qvector q(2);
  init(q);
}

int main() {
  cudaq::sample(10, userKernel, dunkadee);
  std::cout << "Hello, World!\n";
  return 0;
}

// CHECK: Hello, World
