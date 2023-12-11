/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>

struct Sylvester {
  void operator() (int) __qpu__ {}
  void operator() () __qpu__ { // expected-error{{CUDA Quantum kernel class with multiple quantum methods not yet supported}}
     cudaq::qubit q;
     h(q);
  }
};

void foo() {
   Sylvester sylvester;
}
