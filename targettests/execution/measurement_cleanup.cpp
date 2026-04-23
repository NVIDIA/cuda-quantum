/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s
// TODO-FIX-KERNEL-EXEC
// RUN: nvq++ -fkernel-exec-kind=2 --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ int cccx_measure_cleanup() {
  cudaq::qvector qubits(4);
  // Initialize
  x(qubits[0]);
  x(qubits[1]);
  x(qubits[2]);

  // Compute AND-operation
  cudaq::qubit ancilla;
  h(ancilla);
  t(ancilla);
  x<cudaq::ctrl>(qubits[1], ancilla);
  t<cudaq::adj>(ancilla);
  x<cudaq::ctrl>(qubits[0], ancilla);
  t(ancilla);
  x<cudaq::ctrl>(qubits[1], ancilla);
  t<cudaq::adj>(ancilla);
  h(ancilla);
  s<cudaq::adj>(ancilla);

  // Compute output
  x<cudaq::ctrl>(qubits[2], ancilla, qubits[3]);

  // AND's measurement based cleanup.
  bool result = mx(ancilla);
  if (result)
    z<cudaq::ctrl>(qubits[0], qubits[1]);

    return cudaq::to_integer(mz(qubits));
}

int main() {
  auto result = cudaq::run(2, cccx_measure_cleanup);
  for (auto res : result) {
    std::cout << res << '\n';
  }
  return 0;
}

// Expected output is `1111` in binary, which is `15` in decimal
// CHECK: 15
// CHECK: 15
