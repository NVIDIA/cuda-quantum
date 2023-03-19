/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s


#include <cudaq.h>

// CHECK: quake.h (%[[VAL_0:.*]])
// CHECK: %false = arith.constant false
// CHECK: %[[VAL_1:.*]] = memref.alloca() : memref<i1>
// CHECK: memref.store %false, %[[VAL_1]][] : memref<i1>

struct testBoolLiteral {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    h(q);    
    bool bit = false;
    bit = mz(q);
    return bit;
  }
};
