/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: if [ command -v split-file ]; then \
// RUN: split-file %s %t && \
// RUN: nvq++ --enable-mlir -c %t/baselib.cpp -o %t/baselib.o && \
// RUN: nvq++ --enable-mlir -c %t/baseuser.cpp -o %t/baseuser.o && \
// RUN: nvq++ --enable-mlir %t/baselib.o %t/baseuser.o -o %t/base.a.out && \
// RUN: %t/base.a.out | FileCheck %s ; else \
// RUN: echo "skipping"; fi
// clang-format on

//--- baselib.h

#include "cudaq.h"

__qpu__ void dunkadee(cudaq::qvector<> &q);

//--- baselib.cpp

#include "baselib.h"
#include <iostream>

void rollcall() { std::cout << "library function here, sir!\n"; }

__qpu__ void dunkadee(cudaq::qvector<> &q) {
  x(q[0]);
  rollcall();
}

//--- baseuser.cpp

#include "baselib.h"
#include <iostream>

__qpu__ void userKernel(const cudaq::qkernel<void(cudaq::qvector<> &)> &init) {
  cudaq::qvector q(2);
  init(q);
}

int main() {
  userKernel(dunkadee);
  std::cout << "Hello, World!\n";
  return 0;
}

// CHECK: library function here
// CHECK: Hello, World
