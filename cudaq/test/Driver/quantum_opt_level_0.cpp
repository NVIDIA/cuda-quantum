/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Verify that nvq++ compiles -fquantum-opt-level=0 into the kernel launches
// and disables quantum optimization in the AOT code generation.

// RUN: nvq++ -c %s -v -fquantum-opt-level=0 -o %t.o | FileCheck %s

// -fno-quantum-optimization is an alias for -fquantum-opt-level=0.
// RUN: nvq++ -c %s -v -fno-quantum-optimization -o %t.o | FileCheck %s

// CHECK-NOT: cudaq-quake {{.*}}--quantum-opt-level
// CHECK:     cudaq-opt {{.*}}kernel-execution{{[{][^}]*}}quantum-opt-level=0
// CHECK:     cudaq-translate {{.*}}--fdisable-quantum-optimization

#include "cudaq.h"

__qpu__ void kernel() {
  cudaq::qubit q;
  h(q);
}
