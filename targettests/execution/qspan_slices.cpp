/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --emulate %s -o %t --target anyon && %t | FileCheck %s
// RUN: nvq++ --emulate %s -o %t --target anyon --anyon-machine berkeley-25q && %t | FileCheck %s
// RUN: nvq++ --emulate %s -o %t --target ionq && %t | FileCheck %s
// RUN: nvq++ --emulate %s -o %t --target iqm && IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt  %t | FileCheck %s
// RUN: nvq++ --emulate %s -o %t --target oqc && %t | FileCheck %s
// RUN: nvq++ --emulate %s -o %t --target quantinuum && %t | FileCheck %s
// RUN: if %qci_avail; then \
// RUN: nvq++ --emulate %s -o %t --target qci && %t | FileCheck %s; fi

// Tests for --disable-qubit-mapping:
// RUN: nvq++ -v %s -o %t --target oqc --emulate --disable-qubit-mapping && CUDAQ_MLIR_PRINT_EACH_PASS=1 %t |& FileCheck --check-prefix=DISABLE %s
// RUN: nvq++ -v %s -o %t --target iqm --emulate --disable-qubit-mapping && CUDAQ_MLIR_PRINT_EACH_PASS=1 IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt %t |& FileCheck --check-prefix=DISABLE %s
// clang-format on

#include <cudaq.h>
#include <iostream>

__qpu__ void bar(cudaq::qview<> qubits) {
  auto controls = qubits.front(qubits.size() - 1);
  auto &target = qubits.back();
  x<cudaq::ctrl>(controls, target);
}

__qpu__ void foo() {
  cudaq::qvector qubits(4);
  x(qubits);
  bar(qubits);

  auto result = mz(qubits);
}

int main() {
  auto result = cudaq::sample(1000, foo);

#ifndef SYNTAX_CHECK
  std::cout << result.most_probable() << '\n';
  assert("1110" == result.most_probable());
#endif

  return 0;
}

// CHECK: 1110

// For this test, we should see the mapping pass run, but there should be no
// mapping_v2p attribute applied anywhere thereafter.
// DISABLE: IR Dump Before MappingFunc
// DISABLE-NOT: mapping_v2p
