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

// A type that is declared in a kernel is subject to the rules of every type in
// a kernel. This struct has a member function.
struct UsesLocalWithMethod {
  int operator()() __qpu__ {
    // expected-error@+1{{struct with user-defined methods is not allowed in quantum kernel}}
    struct HasMethod {
      int a;
      int get() const { return a; }
    };
    HasMethod h;
    h.a = 1;
    return h.a;
  }
};
// clang-format on
