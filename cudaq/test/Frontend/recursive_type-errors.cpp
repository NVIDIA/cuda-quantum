/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s -verify
// clang-format off

#include "cudaq.h"

// A type that refers to itself cannot be a type in a kernel.
// expected-error@+1{{recursive types are not allowed in kernels}}
struct Node {
  int value;
  Node *next;
};

// Neither can types that refer to each other.
struct Second;
// expected-error@+1{{recursive types are not allowed in kernels}}
struct First {
  Second *other;
};
struct Second {
  First *other;
};

struct UsesNode {
  // expected-error@+1{{failed to generate type for kernel function}}
  void operator()(Node n) __qpu__ { cudaq::qubit q; }
};

struct UsesFirst {
  // expected-error@+1{{failed to generate type for kernel function}}
  void operator()(First f) __qpu__ { cudaq::qubit q; }
};

// clang-format on
