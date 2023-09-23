/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Note: change |& to 2>&1| if running in bash

// First run is to verify compilation errors
// RUN: cudaq-quake %s |& FileCheck --check-prefix COMPILER %s

// Second run is to verify runtime errors for QIR validation
// RUN: nvq++ %s -o %basename_t.x --target quantinuum --emulate 2> /dev/null && ./%basename_t.x |& FileCheck --check-prefix RUNTIME %s

#include <cudaq.h>
#include <iostream>

__qpu__ void init_state() {
  cudaq::qreg<5> q;
  x(q[0]);
  mz(q[99]);
};

int main() {
  auto result = cudaq::sample(1000, init_state);
  for (auto &&[bits, counts] : result) {
    std::cout << bits << '\n';
  }
  return 0;
}

// COMPILER: error: 'quake.extract_ref' op invalid index [99] because >= size [5]
// RUNTIME: qubit [99] is >= required_num_qubits [5]
