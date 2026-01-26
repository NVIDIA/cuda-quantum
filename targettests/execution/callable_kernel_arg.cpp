/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target anyon      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target infleqtion --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target ionq       --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target iqm        --emulate %s -o %t && IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt  %t | FileCheck %s
// RUN: nvq++ --target oqc        --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: if %braket_avail; then nvq++ --target braket --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %quantum_machines_avail; then nvq++ --target quantum_machines --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: nvq++ --enable-mlir %s -o %t
// clang-format on

#include <cudaq.h>
#include <iostream>

__qpu__ void bar(cudaq::qubit &q) { x(q); }

struct baz {
  __qpu__ void operator()(cudaq::qubit &q) { x(q); }
};

struct foo {
  template <typename CallableKernel>
  __qpu__ void operator()(CallableKernel &&func, int size) {
    cudaq::qvector q(size);
    func(q[0]);
    auto result = mz(q[0]);
  }
};

int main() {
  auto result = cudaq::sample(1000, foo{}, baz{}, /*qreg size*/ 1);

#ifndef SYNTAX_CHECK
  std::cout << result.most_probable() << '\n';
  assert("1" == result.most_probable());
#endif

  return 0;
}

// CHECK: 1
