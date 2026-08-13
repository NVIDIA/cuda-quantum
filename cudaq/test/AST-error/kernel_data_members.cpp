/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>

// Error for kernels with data members. Not supported (yet).

// clang-format off
// expected-error@+2{{CUDA-Q kernel class with data members is not yet supported}}
// clang-format on
struct C {
  void operator()(cudaq::qubit &q) __qpu__ { h(q); }

  double unsupported;
};

// expected-error@+1{{class inheritance is not allowed for CUDA-Q kernel}}
struct D : public C {
  void operator()(cudaq::qubit &q) __qpu__ { h(q); }
};
