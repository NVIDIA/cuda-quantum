/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This code is from Issue 296.

// clang-format off
// RUN: if %anyon_avail; then nvq++ --target anyon      --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: nvq++ --target infleqtion --emulate %s -o %t && %t | FileCheck %s
// RUN: if %ionq_avail; then nvq++ --target ionq       --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %iqm_avail; then nvq++ --target iqm        --emulate %s -o %t && IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt  %t | FileCheck %s; fi
// RUN: if %oqc_avail; then nvq++ --target oqc        --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %qbraid_avail; then nvq++ --target qbraid     --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: if %braket_avail; then nvq++ --target braket --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <cudaq.h>
#include <iostream>

__qpu__ void foo(bool value) {
  cudaq::qubit q;
  if (value)
    x(q);

  mz(q);
}

int main() {
  auto result = cudaq::sample(100, foo, true);

#ifndef SYNTAX_CHECK
  std::cout << result.most_probable() << '\n';
  assert("1" == result.most_probable());
#endif

  return 0;
}

// CHECK: 1
