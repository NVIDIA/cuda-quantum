/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std -verify %s

#include <cudaq.h>

// Error for kernels with data members. Not supported (yet).

struct C { // expected-error{{CUDA-Q kernel class with data members is not yet supported}}
  void operator()(cudaq::qubit &q) __qpu__ { h(q); }

  double unsupported;
};

struct D : public C { // expected-error{{class inheritance is not allowed for CUDA-Q kernel}}
  void operator()(cudaq::qubit &q) __qpu__ {
    h(q);
  }
};
