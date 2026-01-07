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
// RUN: nvq++ --enable-mlir -c %t/vector_return_lib.cpp -o %t/vector_return_lib.o && \
// RUN: nvq++ --enable-mlir -c %t/vector_return_main.cpp -o %t/vector_return_main.o && \
// RUN: nvq++ --enable-mlir %t/vector_return_lib.o %t/vector_return_main.o -o %t/vector_return.a.out && \
// RUN: %t/vector_return.a.out | FileCheck %s ; else \
// RUN: echo "skipping" ; fi
// clang-format on

//--- vector_return_lib.h

#pragma once

#include "cudaq.h"

// NB: The __qpu__ here on this declaration cannot be omitted!
__qpu__ std::vector<int> get_a_nice_vector();
void dump_vector_element(int ele);

//--- vector_return_lib.cpp

#include "vector_return_lib.h"

void send_bat_signal() {
  // Signal Gotham
  std::cout << "na na na na na ... BATMAN!\n";
}

__qpu__ std::vector<int> get_a_nice_vector() {
  send_bat_signal();
  return {1, 2, 3, 8, -9};
}

void dump_vector_element(int ele) {
  // print out the element value
  std::cout << "element: " << ele << '\n';
}

//--- vector_return_main.cpp

#include "vector_return_lib.h"

__qpu__ void entry() {
  std::vector<int> v = get_a_nice_vector();
  for (unsigned i = 0; i < v.size(); ++i)
    dump_vector_element(v[i]);
}

int main() {
  // call the entry-point kernel
  entry();
  return 0;
}

// CHECK: na ... BATMAN!
// CHECK: 1
// CHECK: 2
// CHECK: 3
// CHECK: 8
// CHECK: -9

