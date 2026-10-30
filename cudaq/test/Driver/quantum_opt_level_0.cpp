/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Verify that nvq++ forwards -fquantum-opt-level=0 to cudaq-quake.

// RUN: nvq++ -c %s -v -fquantum-opt-level=0 -o %t.o | FileCheck %s

// -fno-quantum-optimization is an alias for -fquantum-opt-level=0.
// RUN: nvq++ -c %s -v -fno-quantum-optimization -o %t.o | FileCheck %s

// CHECK: cudaq-quake {{.*}}--quantum-opt-level=0

#include "cudaq.h"

__qpu__ void kernel() {
  cudaq::qubit q;
  h(q);
}
