/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>

struct Sylvester {
  void operator()(int) __qpu__ {}
  // clang-format off
  // expected-error@+2{{CUDA-Q kernel class with multiple quantum methods not yet supported}}
  // clang-format on
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
  }
};

void foo() { Sylvester sylvester; }
