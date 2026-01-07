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
// RUN: nvq++ --enable-mlir -c %t/anonlib.cpp -o %t/anonlib.o && \
// RUN: nvq++ --enable-mlir -c %t/anonuser.cpp -o %t/anonuser.o && \
// RUN: nvq++ --enable-mlir %t/anonlib.o %t/anonuser.o -o %t/anon.a.out && \
// RUN: %t/anon.a.out | FileCheck %s ; else \
// RUN: echo "skipping" ; fi
// clang-format on

//--- anonlib.h

#include "cudaq.h"

__qpu__ void userKernel(const cudaq::qkernel<void(cudaq::qvector<> &)> &);

//--- anonlib.cpp

#include "anonlib.h"

__qpu__ void userKernel(const cudaq::qkernel<void(cudaq::qvector<> &)> &init) {
  cudaq::qvector q(2);
  init(q);
}

//--- anonuser.cpp

#include "anonlib.h"
#include <iostream>

void rollcall() { std::cout << "elsewhere function here, sir!\n"; }

int main() {
  userKernel([](cudaq::qvector<> &q) __qpu__ {
    x(q[0]);
    rollcall();
  });
  std::cout << "Hello, World!\n";
  return 0;
}

// CHECK: elsewhere function here
// CHECK: Hello, World
