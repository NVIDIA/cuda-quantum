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

enum Color { Red, Green, Blue };

// An enumeration has no representation in a kernel (yet). That is an error that
// says so, not an assertion.
struct UsesEnum {
  int operator()() __qpu__ {
    // expected-error@+2{{statement not supported in qpu kernel}}
    // expected-error@+1{{variable of a type that has no representation in a kernel is not yet supported}}
    Color c = Green;
    // The variable is in scope, but a use of a value that has no representation
    // cannot be lowered either.
    // expected-error@+1{{statement not supported in qpu kernel}}
    return static_cast<int>(c);
  }
};
// clang-format on
