/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: if %anyon_avail; then nvq++ --target anyon      --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: nvq++ --target infleqtion --emulate %s -o %t && %t | FileCheck %s
// RUN: if %ionq_avail; then nvq++ --target ionq       --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %iqm_avail; then nvq++ --target iqm        --emulate %s -o %t && IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt %t | FileCheck %s; fi
// RUN: if %oqc_avail; then nvq++ --target oqc        --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %qbraid_avail; then nvq++ --target qbraid     --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: if %braket_avail; then nvq++ --target braket --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: nvq++ %s -o %t && %t | FileCheck %s
// clang-format on

#include "cudaq.h"
#include <iostream>

int main() {

  auto swapKernel = []() __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    swap(q[0], q[1]);

    mz(q);
  };

  auto counts = cudaq::sample(swapKernel);

#ifndef SYNTAX_CHECK
  std::cout << counts.most_probable() << '\n';
  assert("01" == counts.most_probable());
#endif

  return 0;
}

// CHECK: 01
