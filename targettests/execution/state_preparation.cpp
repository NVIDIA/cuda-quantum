/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Simulators
// RUN: nvq++ %s -o %t && %t | FileCheck %s

// Quantum emulators
// clang-format off
// RUN: nvq++ --target infleqtion --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target ionq       --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target iqm        --emulate %s -o %t && IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt  %t | FileCheck %s
// RUN: nvq++ --target oqc        --emulate %s -o %t && %t | FileCheck %s
// RUN: if %braket_avail; then nvq++ --target braket --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %quantum_machines_avail; then nvq++ --target quantum_machines --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <bitset>
#include <cudaq.h>
#include <iostream>

struct kernel {
  __qpu__ void operator()(std::vector<cudaq::complex> vec) {
    cudaq::qvector qubits{vec};
    mz(qubits);
  }
};

struct kernel2 {
  __qpu__ void operator()(std::vector<cudaq::real> vec) {
    cudaq::qvector qubits{vec};
    mz(qubits);
  }
};

void test1() {
  std::cout << "test1\n";
  std::vector<cudaq::complex> vec{0., 0., 0., 0.};
  for (std::size_t i = 0; i < vec.size(); i++) {
    if (i > 0)
      vec[i - 1] = 0;
    vec[i] = 1;

    auto result = cudaq::sample(kernel{}, vec);

    std::bitset<8> binary(i);
    auto expected = binary.to_string().substr(binary.size() - 2);

    auto bits = result.most_probable();
    std::reverse(bits.begin(), bits.end());
    std::cout << bits << '\n';

    assert(bits == expected);
  }
}

void test2() {
  std::cout << "test2\n";
  std::vector<cudaq::real> vec{0., 0., 0., 0.};
  for (std::size_t i = 0; i < vec.size(); i++) {
    if (i > 0)
      vec[i - 1] = 0;
    vec[i] = 1;

    auto result = cudaq::sample(kernel2{}, vec);

    std::bitset<8> binary(i);
    auto expected = binary.to_string().substr(binary.size() - 2);

    auto bits = result.most_probable();
    std::reverse(bits.begin(), bits.end());
    std::cout << bits << '\n';

    assert(bits == expected);
  }
}

int main() {
  test1();
  test2();
  return 0;
}

// CHECK-LABEL: test1
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK-LABEL: test2
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
