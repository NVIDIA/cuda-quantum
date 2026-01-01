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
// RUN: nvq++ -target quantinuum -emulate -fno-set-target-backend -c %t/emulib.cpp -o %t/emulibx.o && \
// RUN: nvq++ -target quantinuum -emulate -c %t/emuuser.cpp -o %t/emuuserx.o && \
// RUN: nvq++ -target quantinuum -emulate %t/emulibx.o %t/emuuserx.o -o %t/emux.a.out && \
// RUN: %t/emux.a.out | FileCheck %s ; else \
// RUN: echo "skipping" ; fi
// clang-format on

//--- emulib.h

#include "cudaq.h"

__qpu__ void dunkadee(cudaq::qvector<> &q);

//--- emulib.cpp

#include "emulib.h"
#include <iostream>

__qpu__ void dunkadee(cudaq::qvector<> &q) { x(q[0]); }

//--- emuuser.cpp

#include "emulib.h"
#include <iostream>

__qpu__ void userKernel(const cudaq::qkernel<void(cudaq::qvector<> &)> &init) {
  cudaq::qvector q(2);
  init(q);
}

int main() {
  cudaq::sample(10, userKernel,
                cudaq::qkernel<void(cudaq::qvector<> &)>{dunkadee});
  std::cout << "Hello, World!\n";
  return 0;
}

// CHECK: Hello, World
