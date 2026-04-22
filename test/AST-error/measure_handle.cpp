/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

// Verify compile-time diagnostics for `cudaq::measure_handle`:
//
//   1. `operator==` / `operator!=` on `measure_handle` are explicitly deleted
//      (spec §`measure_handle` class). Users must write
//      `discriminate(h1) == discriminate(h2)` instead.
//
//   2. `cudaq::discriminate` on a default-constructed (unbound) handle is
//      rejected by the frontend with the spec-mandated message
//      "discriminating an unbound measure_handle" (spec §`measure_handle`
//      class, unbound-handle concept).

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

// Directly discriminating a default-constructed handle is the canonical
// "unbound" pattern and must be diagnosed.
struct DiscriminateUnbound {
  bool operator()() __qpu__ {
    cudaq::measure_handle h;
    // The primary diagnostic is the unbound-handle error on the
    // `cudaq::discriminate` call; the bridge then fails the enclosing
    // statement because the call did not push a value. Accept both so
    // -verify passes without masking either diagnostic.
    // expected-error@+2{{discriminating an unbound measure_handle}}
    // expected-error@+1{{statement not supported in qpu kernel}}
    return cudaq::discriminate(h);
  }
};
