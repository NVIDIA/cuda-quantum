/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: cudaq-quake %cpp_std %s -verify

#include "cudaq.h"

struct s {
    cudaq::qview<> s;
};
struct test { // expected-error {{recursive quantum struct types are not allowed.}}
  cudaq::qview<> q;
  cudaq::qview<> r;
  s s;
};
__qpu__ void entry_ctor() {
  cudaq::qvector q(2), r(2);
  s s(q);
  test tt(q, r, s); 
  h(tt.r[0]);
}
// expected-error@* {{}}
// expected-error@* {{}}
// expected-error@* {{}}
// expected-error@* {{}}
// expected-error@* {{}}