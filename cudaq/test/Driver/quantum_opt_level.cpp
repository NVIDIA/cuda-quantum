/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Verify that nvq++ records no level by default.

// RUN: nvq++ -c %s -v -o %t.o | FileCheck %s

// CHECK: cudaq-quake
// CHECK-NOT: --quantum-opt-level

#include "cudaq.h"

__qpu__ void kernel() {
  cudaq::qubit q;
  h(q);
}

int main() { return 0; }
