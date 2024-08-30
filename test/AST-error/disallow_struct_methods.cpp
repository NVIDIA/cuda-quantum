/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: cudaq-quake %cpp_std %s -verify

#include "cudaq.h"

struct test { // expected-error {{struct with user-defined methods is not allowed in quantum kernel.}}
  cudaq::qview<> q;
  int myMethod() { return 0; }
};

__qpu__ void kernel() {
  cudaq::qvector q(2);
  test t(q);
}
