/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This code is from Issue 251.

// clang-format off
// RUN: nvq++ --target anyon      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target infleqtion --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target ionq       --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target iqm        --emulate %s -o %t
// RUN: IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt  %t | FileCheck %s
// RUN: IQM_QPU_QA=%iqm_tests_dir/Crystal_20.txt %t | FileCheck %s
// RUN: IQM_QPU_QA=%iqm_tests_dir/Crystal_54.txt %t | FileCheck %s
// RUN: nvq++ --target oqc        --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: if %braket_avail; then nvq++ --target braket --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: nvq++ --enable-mlir %s -o %t
// RUN: cudaq-quake %s | cudaq-opt --promote-qubit-allocation | FileCheck --check-prefixes=MLIR %s

#include <cudaq.h>
#include <iostream>

struct simple_x {
  void operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
    mz(q);
  }
};

// MLIR-LABEL:   func.func @__nvqpp__mlirgen__simple_x()
// MLIR-NOT:       quake.alloca !quake.ref
// MLIR:           %[[VAL_0:.*]] = quake.alloca !quake.veq<1>
// MLIR-NEXT:      %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<1>) -> !quake.ref

int main() {
  auto result = cudaq::sample(simple_x{});

#ifndef SYNTAX_CHECK
  std::cout << result.most_probable() << '\n';
  assert("1" == result.most_probable());
#endif

  return 0;
}

// CHECK: 1
