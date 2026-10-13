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

// A construct that the bridge does not know about is an error. It must not be
// ignored, which would give a kernel that is not the one that was written.

struct UsesNew {
  void operator()() __qpu__ {
    cudaq::qubit q;
    // expected-error@+2{{statement not supported in qpu kernel}}
    // expected-error@+1{{'CXXNewExpr' is not yet supported in a kernel}}
    int *p = new int(3);
    // The variable that could not be declared is still in scope, so these are
    // not errors too.
    int k = *p;
    if (*p > 2)
      x(q);
  }
};

struct UsesThrow {
  void operator()() __qpu__ {
    cudaq::qubit q;
    // expected-error@+2{{statement not supported in qpu kernel}}
    // expected-error@+1{{'CXXThrowExpr' is not yet supported in a kernel}}
    throw 1;
  }
};
// clang-format on
