/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

// Verify that `cudaq::measure_handle` rejects direct equality / inequality
// comparisons at compile time. The spec (§`measure_handle` class) deletes
// `operator==` and `operator!=` on the type so users cannot conjure a bool
// without an explicit `cudaq::discriminate`; `discriminate(h1) ==
// discriminate(h2)` is the intended replacement.

#include <cudaq.h>

// expected-note@* 0+ {{}}

struct EqOnHandle {
  void operator()() __qpu__ {
    cudaq::measure_handle h1;
    cudaq::measure_handle h2;
    // expected-error@+1{{deleted}}
    bool b = (h1 == h2);
    (void)b;
  }
};

struct NeOnHandle {
  void operator()() __qpu__ {
    cudaq::measure_handle h1;
    cudaq::measure_handle h2;
    // expected-error@+1{{deleted}}
    bool b = (h1 != h2);
    (void)b;
  }
};
