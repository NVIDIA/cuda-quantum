/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

__qpu__ void qernel() {
   cudaq::qvector qv(2);
   h(qv);
   x<cudaq::ctrl>(qv[0], qv[1]);
}

// CHECK-LABEL:  module
// CHECK-SAME:   cc.sizeof_string = {{[0-9]+}} : i64
