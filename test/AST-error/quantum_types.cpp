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
  void operator()(cudaq::qspan<> q) { // expected-error{{may not use quantum types in non-kernel functions}}
    mz(q);
  }
};

struct S2 {
  void operator()() {
    cudaq::qubit b; // expected-error{{may not use quantum types in non-kernel functions}}
    mz(b);
  }
};

struct S3 {
  void operator()() {
    cudaq::qreg<4> r; // expected-error{{may not use quantum types in non-kernel functions}}
    mz(r);
  }
};

struct S4 {
  void operator()(cudaq::qreg<4> r) { // expected-error{{may not use quantum types in non-kernel functions}}
    mz(r);
  }
};

struct S5 {
  void operator()() {
     cudaq::qreg r(10); // expected-error{{may not use quantum types in non-kernel functions}}
    mz(r);
  }
};
