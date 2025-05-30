/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// clang-format off
// RUN: nvq++ %cpp_std --target infleqtion      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target anyon                    --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target ionq                     --emulate %s -o %t && %t | FileCheck %s
// 2 different IQM machines for 2 different topologies
// RUN: nvq++ --target iqm --iqm-machine Adonis --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target iqm --iqm-machine Apollo --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target oqc                      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target quantinuum               --emulate %s -o %t && %t | FileCheck %s
// RUN: if %braket_avail; then nvq++ --target braket --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <cudaq.h>
#include <cudaq/algorithm.h>

// The example here shows a simple use case for the `cudaq::observe`
// function in computing expected values of provided spin_ops.

struct kernel {
  auto operator()(int n) __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    for (int i = 0; i < n; i++) {
    switch (n)
    {
    case 1:
      /* code */
      x(q[0]);
      break;

    case 2:
      /* code */
      x(q[2]);
      break;
    
    default:
      break;
    }
  }
  }
};

int main() {

  auto results = cudaq::sample(kernel{}, 3);
  results.dump();
  return 0;
}

