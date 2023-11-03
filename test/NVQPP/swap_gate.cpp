/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target ionq                     --emulate %s -o %basename_t.x && ./%basename_t.x | FileCheck %s
// RUN: nvq++ --target iqm --iqm-machine Adonis --emulate %s -o %basename_t.x && ./%basename_t.x | FileCheck %s
// RUN: nvq++ --target oqc                      --emulate %s -o %basename_t.x && ./%basename_t.x | FileCheck %s
// RUN: nvq++ --target quantinuum               --emulate %s -o %basename_t.x && ./%basename_t.x | FileCheck %s

#include "cudaq.h"
#include <iostream>

int main() {

  auto swapKernel = []() __qpu__ {
    cudaq::qreg q(2);
    x(q[0]);
    swap(q[0], q[1]);

    auto result = mz(q);
  };

  auto counts = cudaq::sample(swapKernel);

#ifndef SYNTAX_CHECK
  std::cout << counts.most_probable() << '\n';
  assert("01" == counts.most_probable());
#endif

  return 0;
}

// CHECK: 01
