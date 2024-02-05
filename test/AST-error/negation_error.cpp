/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// The error messages produced by a compiler using the -std=c++17 option cannot
// be expected to be the same.

// REQUIRES: c++20
// RUN: cudaq-quake -verify %s

#include <cudaq.h>

struct NegationOperatorTest {
  void operator()() __qpu__ {
    cudaq::qvector qr(3);
     // expected-error@+1{{target qubit cannot be negated}}
    x<cudaq::ctrl>(qr[0], qr[1], !qr[2]);
    rz(2.0, !qr[0]); // expected-error{{target qubit cannot be negated}}
  }
};
