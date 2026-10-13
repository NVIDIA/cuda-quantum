/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s
// clang-format off

#include "cudaq.h"

// An asm statement cannot be lowered, and it must not be silently dropped.
struct UsesAsm {
  void operator()() __qpu__ {
    cudaq::qubit q;
    // expected-error@+2{{statement not supported in qpu kernel}}
    // expected-error@+1{{asm statement is not yet supported}}
    asm volatile("nop");
  }
};
// clang-format on
