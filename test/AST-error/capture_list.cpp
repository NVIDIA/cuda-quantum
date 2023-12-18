/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>

// Test that we raise an error for lambda's with capture lists.

struct D {
  template <typename KERNEL>
  void operator() (KERNEL &&qernel) __qpu__ {
    cudaq::qubit q;
    qernel(q);
    mz(q);
  }
};

struct LambdaCaptureList {
  void operator() () __qpu__ {
     std::size_t i = 42;
     D{}([i](cudaq::qubit& q) { h(q); }); // expected-error{{lambda expression with explicit captures is not yet supported}}
  }
};
