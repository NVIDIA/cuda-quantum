/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>

struct S1 {
  // expected-error@+1{{may not use quantum types in non-kernel functions}}
  void operator()(cudaq::qview<> q) {
    mz(q);
  }
};

struct S2 {
  void operator()() {
    // expected-error@+1{{may not use quantum types in non-kernel functions}}
    cudaq::qubit b;
    mz(b);
  }
};

struct S3 {
  void operator()() {
    // expected-error@+1{{may not use quantum types in non-kernel functions}}
    cudaq::qarray<4> r;
    mz(r);
  }
};

struct S4 {
  // expected-error@+1{{may not use quantum types in non-kernel functions}}
  void operator()(cudaq::qarray<4> r) {
    mz(r);
  }
};

struct S5 {
  void operator()() {
    // expected-error@+1{{may not use quantum types in non-kernel functions}}
    cudaq::qvector r(10);
    mz(r);
  }
};

