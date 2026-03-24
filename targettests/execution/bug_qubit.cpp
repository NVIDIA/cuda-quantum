/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This code is from Issue 251.

// clang-format off
// RUN: nvq++ --target anyon      --emulate %s -o %t && %t
// RUN: nvq++ --target infleqtion --emulate %s -o %t && %t
// RUN: nvq++ --target ionq       --emulate %s -o %t && %t
// RUN: nvq++ --target iqm        --emulate %s -o %t
// RUN: IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt  %t
// RUN: IQM_QPU_QA=%iqm_tests_dir/Crystal_20.txt %t
// RUN: IQM_QPU_QA=%iqm_tests_dir/Crystal_54.txt %t
// RUN: nvq++ --target oqc        --emulate %s -o %t && %t
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t
// RUN: if %braket_avail; then nvq++ --target braket --emulate %s -o %t && %t; fi
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t; fi
// RUN: cudaq-quake %s | cudaq-opt --promote-qubit-allocation | FileCheck --check-prefixes=MLIR %s
// clang-format on

#include <cudaq.h>
#include <iostream>

struct simple_x {
  void operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
    mz(q);
  }
};

// clang-format on
// MLIR-LABEL:   func.func @__nvqpp__mlirgen__simple_x()
// MLIR-NOT:       quake.alloca !quake.ref
// MLIR:           %[[VAL_0:.*]] = quake.alloca !quake.veq<1>
// MLIR-NEXT:      %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<1>) -> !quake.ref
// clang-format off

int main() {
  auto result = cudaq::sample(simple_x{});

#ifndef SYNTAX_CHECK
  std::cout << result.most_probable() << '\n';
  // Success is "1".
  return std::string{"1"} != result.most_probable();
#endif

  return 0;
}
