/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Note: change |& to 2>&1 if running in bash
// RUN: nvq++ -v %s -o %t --target oqc --emulate && %t |& FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ bool init_state() {
  cudaq::qubit q;
  x(q);
  auto result = mz(q);
  x(q);   // base profile does not allow operations after measurements
  return mz(q);
};

int main() {
  auto result = cudaq::run(1000, init_state);
  return 0;
}

// Note: Need to support `run` with Base profile so as to enable this check
// XCHECK: reversible function __quantum__qis__x__body came after irreversible function
// CHECK: `run` is not yet supported on this target.