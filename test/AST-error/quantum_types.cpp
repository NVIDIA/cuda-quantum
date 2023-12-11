/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>

struct S1 {
  // expected-warning@+3{{}}
  // expected-note@* {{}}
  // expected-error@+1{{may not use quantum types in non-kernel functions}}
  void operator()(cudaq::qspan<> q) {
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
    // expected-warning@+3{{}}
    // expected-note@* {{}}
    // expected-error@+1{{may not use quantum types in non-kernel functions}}
    cudaq::qreg<4> r;
    mz(r);
  }
};

struct S4 {
  // expected-warning@+3{{}}
  // expected-note@* {{}}
  // expected-error@+1{{may not use quantum types in non-kernel functions}}
  void operator()(cudaq::qreg<4> r) {
    mz(r);
  }
};

struct S5 {
  void operator()() {
    // expected-error@+1{{may not use quantum types in non-kernel functions}}
    cudaq::qreg r(10);
    mz(r);
  }
};

