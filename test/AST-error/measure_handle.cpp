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
//   1. Reaching any bool-coercion context (`if`, `while`, ternary, `!`,
//      `&&`, `||`, `==`, `!=`, `bool b = ...`, `return ...`, `bool(...)`,
//      `static_cast<bool>(...)`, `assert`, a function argument requiring
//      `measure_handle -> bool`, ...) with a default-constructed (unbound)
//      handle is rejected by the frontend with the spec-mandated message
//      "discriminating an unbound measure_handle" (spec `measure_handle`,
//      Operational Semantics and unbound-handle concept). The bridge
//      emits `quake.discriminate` at the coercion site and raises the
//      diagnostic when the source operand is still a default-constructed
//      handle at that point.
//
//   2. Entry-point kernels may not name `measure_handle` -- directly or via
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

// Reaching an implicit bool-coercion context (`if`) with a
// default-constructed handle is the canonical "unbound" pattern and must
// be diagnosed.
struct DiscriminateUnbound {
  bool operator()() __qpu__ {
    cudaq::measure_handle h;
    // expected-error@+1{{discriminating an unbound measure_handle}}
    if (h)
      return true;
    return false;
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
    return mz(q);
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

struct BoundaryReferenceParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(cudaq::measure_handle &h) __qpu__ { (void)h; }
};

// User-defined aggregate struct wrapping `measure_handle`. Exercises the
// `cc::StructType` recursion branch of `containsMeasureHandle` via a named
// record rather than a `std::tuple` / `std::pair` specialization.
struct MeasureHandleHolder {
  cudaq::measure_handle h;
};

struct BoundaryAggregateParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(MeasureHandleHolder s) __qpu__ { (void)s; }
};

// Nested container: `std::pair<int, std::vector<measure_handle>>`. Pair of
// `int` + stdvec-of-handle exercises two recursion steps of
// `cudaq::cc::containsMeasureHandle`: `cc.struct` -> `cc.stdvec` ->
// `!cc.measure_handle`. The existing `BoundaryVectorParam`,
// `BoundaryTupleParam`, `BoundaryPairParam`, and `BoundaryPointerParam`
// cases already cover the single-step recursion for each container shape;
// this case proves the walk is genuinely recursive rather than
// one-level-deep.
struct BoundaryPairOfVectorParam {
  // expected-error@+1{{measure_handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::pair<int, std::vector<cudaq::measure_handle>> p) __qpu__ {
    (void)p;
  }
};

// FIXME(measure_handle follow-up): the following container shapes are
// listed in the spec's Kernel Signature Rule but trip a pre-existing
// `getWidthAndAlignment` assertion inside the C++ -> MLIR type mapper
// before `hasMeasureHandleInSignature` can run, so the boundary diagnostic
// cannot fire and the bridge aborts with an LLVM signal instead:
//
//   - `std::array<cudaq::measure_handle, N>`
//   - `std::vector<std::tuple<..., cudaq::measure_handle, ...>>`
//   - `std::optional<cudaq::measure_handle>`
//   - `std::variant<..., cudaq::measure_handle, ...>`
//
// The same type mapper also aborts on `VisitInitListExpr` for a
// `return {};` in a kernel whose return type transitively names
// `measure_handle` (e.g. `std::vector<measure_handle>` as a return type),
// which is why only `BoundaryDirectReturn` (body uses `return mz(q);`)
// currently appears here. Once the type mapper learns a fallback for
// opaque classical handles (teach `getWidthAndAlignment` to treat
// `MeasureHandleType` members as size/alignment of `i64`, or short-circuit
// layout computation for signatures already flagged by the boundary
// walker), add explicit return-position cases for every container shape
// listed above.
