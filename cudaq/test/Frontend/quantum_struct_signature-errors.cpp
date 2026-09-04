/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s -verify

#include "cudaq.h"

struct test {
  cudaq::qubit &r;
  cudaq::qview<> q;
};

// expected-error@+1 {{kernel result type not supported}}
__qpu__ test kernel(cudaq::qubit &q, cudaq::qview<> qq) {
  test result(q, qq);
  return result;
}
