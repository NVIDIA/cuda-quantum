/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s -verify

#include "cudaq.h"

// expected-error@+1 {{struct with user-defined methods is not allowed}}
struct test {
  cudaq::qview<> q;
  int myMethod() { return 0; }
};

__qpu__ void kernel() {
  cudaq::qvector q(2);
  test t(q);
}
