/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Consecutive `if`s on one integer are folded into a `switch` by the LLVM
// optimizer, which the adaptive profile does not allow.

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --add-dealloc --expand-measurements --factor-quantum-alloc --expand-control-veqs --cc-loop-unroll --canonicalize --multicontrol-decomposition --lower-to-cfg --cse --combine-quantum-alloc --canonicalize | cudaq-translate --convert-to=qir-adaptive:1.0:int_computations | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --add-dealloc --expand-measurements --factor-quantum-alloc --expand-control-veqs --cc-loop-unroll --canonicalize --multicontrol-decomposition --lower-to-cfg --cse --combine-quantum-alloc --canonicalize | cudaq-translate --convert-to=qir-adaptive:1.0:int_computations,allow_all_instructions | FileCheck %s --check-prefix=ALLOW
// clang-format on

#include "cudaq/qis/qubit_qis.h"

struct syndrome_lookup {
  bool operator()() __qpu__ {
    cudaq::qvector q(4);
    x(q[0]);
    x(q[2]);
    bool m0 = mz(q[0]);
    bool m1 = mz(q[1]);
    bool m2 = mz(q[2]);
    int code = (m0 ? 1 : 0) | (m1 ? 2 : 0) | (m2 ? 4 : 0);
    if (code == 5)
      x(q[3]);
    if (code == 3)
      z(q[3]);
    return mz(q[3]);
  }
};

// CHECK-LABEL: define i1 @__nvqpp__mlirgen__syndrome_lookup()
// CHECK: icmp
// CHECK-NOT: switch
// CHECK: ret i1

// ALLOW-LABEL: define i1 @__nvqpp__mlirgen__syndrome_lookup()
// ALLOW: switch
