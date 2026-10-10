/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s
// clang-format off

#include <cudaq.h>

struct not_a_block {
  void operator()() __qpu__ {
    cudaq::qubit q;
    // expected-error@+2{{statement not supported in qpu kernel}}
    // expected-error@+1{{an atomic quantum region must be a compound statement}}
    [[cudaq::atomic_region]] x(q);
  }
};

// The attribute applies to functions, not to other declarations.
// expected-warning@+1{{attribute only applies to functions}}
[[cudaq::atomic_region]] int not_a_function = 0;
// clang-format on
