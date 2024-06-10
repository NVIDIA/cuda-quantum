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

void bell_pair() __qpu__ {
  cudaq::qvector q(2);
  cudaq::h(q[0]);
  cudaq::x<cudaq::ctrl>(q[0], q[1]);
  cudaq::mz(q);
}

int main() {
  auto result = cudaq::sample(1000, bell_pair);
  result.dump();
  std::cout << cudaq::draw(bell_pair) << std::endl;

  std::cout << cudaq::get_quake("bell_pair") << std::endl;

  //std::cout << cudaq::getQIR("bell_pair") << std::endl;

  // New: convert to a format (qir, qir-base, qir-adaptive, openqasm, etc)
 // std::cout << cudaq::translate(bell_pair, "openqasm") << std::endl;
  return 0;
}

// tools:

// cudaq-quake targettests/execution/program.cpp | cudaq-opt --canonicalize | cudaq-translate --convert-to=openqasm
