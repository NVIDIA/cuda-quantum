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

// expected-error@+1{{recursive types are not allowed in kernels}}
struct Node {
  int value;
  Node *next;
};

// A variable that cannot be declared is still in scope: the uses of it are not
// errors too.
struct UsesLocal {
  void operator()() __qpu__ {
    cudaq::qubit q;
    // expected-error@+1{{statement not supported in qpu kernel}}
    Node n;
    n.value = 1;
    int copy = n.value + 2;
  }
};
// clang-format on
