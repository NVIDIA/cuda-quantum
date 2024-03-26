/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | FileCheck %s
// XFAIL: *
// [SKIP_TEST]: Not implemented

#include <cudaq.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

/// Temporary placeholder of macro
#define cudaq_register_op(NAME, DATA)
void NAME(Qubits... q) {
  // process
}

cudaq_register_op("custom_h",
                  {{M_SQRT1_2, M_SQRT1_2}, {M_SQRT1_2, -M_SQRT1_2}});
cudaq_register_op("custom_x", {{0, 1}, {1, 0}});

void custom_operation() __qpu__ {
  cudaq::qvector qubits(2);
  custom_h(qubits[0]);
  custom_x.ctrl(qubits[0], qubits[1]);
}

int main() {
  auto result = cudaq::sample(custom_operation);
  std::cout << result.most_probable() << '\n';
  return 0;
}
