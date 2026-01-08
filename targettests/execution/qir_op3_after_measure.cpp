/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Note: change |& to 2>&1 if running in bash
// RUN: nvq++ %s -o %t --target oqc --emulate && %t |& FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ std::vector<bool> init_state() {
  cudaq::qubit q0;
  cudaq::qubit q1;
  x(q0);
  mz(q0);
  // The Base Profile spec technically allows a measurement like this because it
  // isn't operating on an already-measured qubit, but it requires that the
  // compiler to reorder the q1 operation to be before the q0 measurement.
  x(q1);
  return {mz(q0), mz(q1)};
};

int main() {
  auto result = cudaq::run(100, init_state);
  return 0;
}

// Note: Need to support `run` with Base profile so as to enable this check
// XCHECK-NOT: reversible function __quantum__qis__x__body came after irreversible function
// CHECK: `run` is not yet supported on this target.
