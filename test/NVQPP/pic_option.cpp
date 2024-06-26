/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: nvq++ --enable-mlir -c %s -v -fPIC -o %t

#include "cudaq.h"

__qpu__ void bell() {
    cudaq::qubit q, r;
    h(q);
    x<cudaq::ctrl>(q,r);
}
