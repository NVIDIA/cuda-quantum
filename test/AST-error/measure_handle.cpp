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
//
//   3. Entry-point kernels may not name `measure_handle` -- directly or via
//      any container (`std::vector`, `std::tuple`, `std::pair`, pointers, ...)
//      -- in any parameter or return position. The diagnostic is the spec's
//      exact message "measure_handle cannot cross the host-device boundary;
//      entry-point kernels must discriminate first" (spec §Kernel Signature
//      Rule). Pure-device kernels (those whose signatures already include a
//      qubit-typed argument and so are not classified as entry points) are
//      unrestricted; that case is exercised in the AST-Quake tests.

#include <cudaq.h>
#include <tuple>
#include <utility>
#include <vector>

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

// Host-device boundary diagnostics. Each kernel below classifies as an
// entry point (no qubit-typed argument) and must therefore be rejected.
// The recursive type walk in `ASTBridge.cpp::hasMeasureHandleInSignature`
// covers each container shape generically; one diagnostic per kernel is
// emitted by `cudaq::details::reportClangError`.

struct BoundaryDirectParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(cudaq::measure_handle h) __qpu__ { (void)h; }
};

struct BoundaryDirectReturn {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  cudaq::measure_handle operator()() __qpu__ {
    cudaq::qubit q;
    return mz_handle(q);
  }
};

struct BoundaryVectorParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::vector<cudaq::measure_handle> h) __qpu__ { (void)h; }
};

struct BoundaryTupleParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::tuple<int, cudaq::measure_handle> h) __qpu__ {
    (void)h;
  }
};

struct BoundaryPairParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::pair<bool, cudaq::measure_handle> h) __qpu__ {
    (void)h;
  }
};

struct BoundaryPointerParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(cudaq::measure_handle *h) __qpu__ { (void)h; }
};
