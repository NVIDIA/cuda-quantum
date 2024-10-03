/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: cudaq-quake %s -verify

#include "cudaq.h"

// expected-error@+1 {{quantum struct has invalid member type}}
struct error1 {
  cudaq::qvector<4> wrong;
};

__qpu__ void bug1(error1&);

// expected-error@+1 {{quantum struct has invalid member type}}
struct error2 {
  cudaq::qubit cubit;
};

__qpu__ void bug2(error2&);

// expected-error@+2 {{quantum struct has invalid member type}}
// expected-error@+1 {{quantum struct has invalid member type}}
struct error3 {
  cudaq::qubit nope;
  cudaq::qvector<2> sorry;
};

__qpu__ void bug3(error3&);

__qpu__ void funny() {
   error1 e1;
   error2 e2;
   error3 e3;
   bug1(e1);
   bug2(e2);
   bug3(e3);
}
