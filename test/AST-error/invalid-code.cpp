/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: ( cudaq-quake %s || true ) 2>&1 | FileCheck %s

// CHECK: C++ source has errors. nvq++ cannot proceed.

#include <cudaq.h>

__qpu__ void invalid_code() {
  cudaq::qubit reg(4);
  for (size_t i = 0; i < reg.size(); ++i) {
    x(reg[i]);
  }
}
