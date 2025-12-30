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
// RUN: nvq++ --enable-mlir -c %t/classlib.cpp -o %t/classlib.o && \
// RUN: nvq++ --enable-mlir -c %t/classuser.cpp -o %t/classuser.o && \
// RUN: nvq++ --enable-mlir %t/classlib.o %t/classuser.o -o %t/class.a.out && \
// RUN: %t/class.a.out | FileCheck %s ; else \
// RUN: echo "skipping" ; fi
// clang-format on

//--- classlib.h

#include "cudaq.h"

struct HereIsTheThing {
  void operator()(cudaq::qvector<> &q) __qpu__;
};

//--- classlib.cpp

#include "classlib.h"
#include <iostream>

void rollcall() { std::cout << "library function here, sir!\n"; }

void HereIsTheThing::operator()(cudaq::qvector<> &q) __qpu__ {
  x(q[0]);
  rollcall();
}

//--- classuser.cpp

#include "classlib.h"
#include <iostream>

__qpu__ void userKernel(const cudaq::qkernel<void(cudaq::qvector<> &)> &init) {
  cudaq::qvector q(2);
  init(q);
}

int main() {
  userKernel(HereIsTheThing{});
  std::cout << "Hello, World!\n";
  return 0;
}

// CHECK: library function here
// CHECK: Hello, World
