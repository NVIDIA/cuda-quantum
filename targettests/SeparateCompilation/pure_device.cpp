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
// RUN: nvq++ --enable-mlir -c %t/pd_lib.cpp -o %t/pd_lib.o && \
// RUN: nvq++ --enable-mlir -c %t/pd_main.cpp -o %t/pd_main.o && \
// RUN: nvq++ --enable-mlir %t/pd_lib.o %t/pd_main.o -o %t/pd.a.out && \
// RUN: %t/pd.a.out | FileCheck %s ; else \
// RUN: echo "skipping" ; fi
// clang-format on

//--- pd_lib.h

#pragma once

#include "cudaq.h"

// NB: The __qpu__ here on this declaration cannot be omitted!
__qpu__ void callMe(cudaq::qvector<> &q, int i);

//--- pd_lib.cpp

#include "pd_lib.h"

void send_bat_signal() { std::cout << "na na na na na ... BATMAN!\n"; }

__qpu__ void callMe(cudaq::qvector<> &q, int i) {
  ry(2.2, q[0]);
  send_bat_signal();
}

//--- pd_main.cpp

#include "pd_lib.h"

__qpu__ void entry() {
  cudaq::qvector q(2);
  callMe(q, 5);
}

int main() {
  entry();
  return 0;
}

// CHECK: na ... BATMAN!
