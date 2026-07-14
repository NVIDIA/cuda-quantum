/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: if %oqc_avail; then nvq++ --target oqc --emulate %s -o %t && %t; fi
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t; fi
// RUN: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t
// clang-format on

#include <cudaq.h>

__qpu__ void reordered_terminal_measurements() {
  cudaq::qvector qubits(3);
  x(qubits[0]);
  x(qubits[2]);

  // Measure out of allocation order so the provider reports compact QIR
  // results as "110" while CUDA-Q must present the global result as "101".
  mz(qubits[2]);
  mz(qubits[0]);
  mz(qubits[1]);
}

int main() {
  const auto result = cudaq::sample(100, reordered_terminal_measurements);

  // Default sampling follows the allocation-order convention, as validated in
  // `test_explicit_measurements.py::test_measurement_order`.
  return result.most_probable() == "101" ? 0 : 1;
}
