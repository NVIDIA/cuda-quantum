/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s -verify

#include "cudaq.h"

struct s {
    cudaq::qview<> s;
};
// expected-error@+2{{recursive quantum struct types are not allowed}}
// expected-error@+1{{quantum struct has invalid member type}}
struct test {
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
