/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std --target ionq                     --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Adonis --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target oqc                      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target quantinuum               --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ -std=c++17 --enable-mlir %s -o %t
// clang-format on

#include <cudaq.h>
#include <cudaq/algorithms/draw.h>
#include <iostream>

// @cudaq.kernel
// def bell_pair():
//       q = cudaq.qvector(2)
//       h(q[0])
//       cx(q[0], q[1])
//       mz(q)

// print(bell_pair())

void bell_pair() __qpu__ {
  cudaq::qvector q(2);
  cudaq::h(q[0]);
  cudaq::x<cudaq::ctrl>(q[0], q[1]);
  cudaq::mz(q);


  // cudaq::qubit qubit2;
  // cudaq::y(qubit2);
  // mz(qubit2);

  // cudaq::qubit qubit3;
  // cudaq::z(qubit3);
  // mz(qubit3);
}

int main() {
  //cudaq::set_target_backend("nvidia");
  auto result = cudaq::sample(1000, variable_qreg);
  result.dump();
  std::cout << cudaq::draw(variable_qreg) << std::endl;

  // New: convert to a format (qir, qir-base, qir-adaptive, qasm, etc)
  // std::cout << cudaq::convert(variable_qreg{}, "format") << std::endl;
  return 0;
}

// tools:

// cudaq-quake targettests/execution/program.cpp | cudaq-opt --canonicalize | cudaq-translate --convert-to=openqasm
