/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// Simulators
// RUN: nvq++ %cpp_std --enable-mlir  %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --library-mode %s -o %t && %t | FileCheck %s

// Quantum emulators
// RUN: nvq++ %cpp_std --target quantinuum               --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target ionq                     --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target anyon                    --emulate %s -o %t && %t | FileCheck %s
// 2 different IQM machines for 2 different topologies
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Adonis --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Apollo --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target oqc                      --emulate %s -o %t && %t | FileCheck %s
// RUN: if %braket_avail; then nvq++ %cpp_std --target braket --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <cudaq.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cudaq/algorithms/draw.h>

using namespace std::complex_literals;
using complex = std::complex<cudaq::real>;

void kernel(int qubit_count) __qpu__ {
  cudaq::qvector qvector(qubit_count);
  // 2-qubit GHZ state.
  h(qvector[0]);
  for (auto qubit : cudaq::range(qubit_count - 1)) {
    x<cudaq::ctrl>(qvector[qubit], qvector[qubit + 1]);
  }
  // If we do not specify measurements, all qubits are measured in
  // the Z-basis by default or we can manually specify it also
//   mz(qvector);
}

void printCounts(cudaq::sample_result &result) {
    std::vector<std::string> values{};
    for (auto &&[bits, counts] : result) {
        values.push_back(bits);
    }
    
    std::sort(values.begin(), values.end());
    for (auto &&bits : values) {
        std::cout << bits << std::endl;
    }
}
    
int main() {
  int qubit_count = 2;
  {
    std::cout << "Executing kernel - sample" << std::endl;
    auto produced_str = cudaq::draw(kernel, qubit_count);
    auto counts = cudaq::sample(kernel, qubit_count);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Executing kernel - sample
  // CHECK: 00
  // CHECK: 11
  // clang-format on

  {
    std::cout << "Executing kernel - observe" << std::endl;
    // Define a Hamiltonian in terms of Pauli Spin operators.
    auto hamiltonian = cudaq::spin_op::z(0) + cudaq::spin_op::y(1) +
    cudaq::spin_op::x(0) * cudaq::spin_op::z(0);

    // Compute the expectation value given the state prepared by the kernel.
    auto counts = cudaq::observe(kernel, hamiltonian, qubit_count).expectation();
    std::cout << counts << std::endl;

    // Compute the statevector of the kernel
    cudaq::state t = cudaq::get_state(kernel, qubit_count);
  }
  // clang-format off
  // CHECK: Executing kernel - observe
  // CHECK: 1
  // clang-format on
}
