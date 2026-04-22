/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// Exercise the C++ AST bridge for the `cudaq::measure_handle` API defined in
// the measure_handle spec:
//
//   * `cudaq::measure_handle`                 => `!cc.measure_handle`
//   * `mz_handle(qubit&)`                      => `quake.mz ... -> !cc.measure_handle`
//   * `mz_handle(qvector)` (range overload)    => `quake.mz ... -> !cc.stdvec<!cc.measure_handle>`
//   * `cudaq::discriminate(handle)`            => `quake.discriminate ... -> i1`
//   * `cudaq::discriminate(vector<handle>)`    => `quake.discriminate ... -> !cc.stdvec<i1>`
//   * `cudaq::to_integer(vector<handle>)`      => `quake.discriminate` + `__nvqpp_cudaqConvertToInteger`
//
// Importantly, the handle-returning measurement overloads must *not* inline a
// `quake.discriminate`; only explicit `cudaq::discriminate` calls do.

#include <cudaq.h>

// 1. Scalar handle: `mz_handle(q)` emits `quake.mz ... -> !cc.measure_handle`
//    with no follow-on `quake.discriminate`.
struct ScalarHandle {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle h = mz_handle(q);
    (void)h;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ScalarHandle()
// CHECK:           %[[VAL_Q:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_H:.*]] = quake.mz %[[VAL_Q]] : (!quake.ref) -> !cc.measure_handle
// CHECK-NOT:       quake.discriminate
// CHECK:           return
// CHECK:         }

// 2. Explicit discrimination of a scalar handle lowers to `quake.discriminate`.
struct ScalarDiscriminate {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    auto h = mz_handle(q);
    return cudaq::discriminate(h);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ScalarDiscriminate() -> i1
// CHECK:           %[[VAL_Q:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_Q]] : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_B:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           return %[[VAL_B]] : i1
// CHECK:         }

// 3. Register-form `mz_handle(qvec)` emits a stdvec of handles, again without
//    an inlined discriminate.
struct RegisterHandle {
  void operator()() __qpu__ {
    cudaq::qvector v(4);
    std::vector<cudaq::measure_handle> hs = mz_handle(v);
    (void)hs;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__RegisterHandle()
// CHECK:           %[[VAL_V:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %{{.*}} = quake.mz %[[VAL_V]] : (!quake.veq<4>) -> !cc.stdvec<!cc.measure_handle>
// CHECK-NOT:       quake.discriminate
// CHECK:           return
// CHECK:         }

// 4. Vector form of `cudaq::discriminate` lowers to a stdvec `quake.discriminate`.
//    The result is consumed via `cudaq::to_integer` so the discriminate op is
//    not dead-code-eliminated by the pure-op DCE that runs in cudaq-quake.
struct RegisterDiscriminate {
  std::int64_t operator()() __qpu__ {
    cudaq::qvector v(4);
    auto hs = mz_handle(v);
    std::vector<bool> bits = cudaq::discriminate(hs);
    return cudaq::to_integer(bits);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__RegisterDiscriminate() -> i64
// CHECK:           %[[VAL_V:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_V]]{{.*}}: (!quake.veq<4>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %{{.*}} = quake.discriminate %[[VAL_M]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.stdvec<i1>
// CHECK:           return %{{.*}} : i64
// CHECK:         }

// 5. `cudaq::to_integer(std::vector<measure_handle>)` emits a stdvec
//    discriminate followed by the bit-packing intrinsic
//    `__nvqpp_cudaqConvertToInteger`.
struct HandleToInteger {
  std::int64_t operator()() __qpu__ {
    cudaq::qvector v(4);
    auto hs = mz_handle(v);
    return cudaq::to_integer(hs);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__HandleToInteger() -> i64
// CHECK:           quake.mz %{{.*}} : (!quake.veq<4>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[VAL_BITS:.*]] = quake.discriminate %{{.*}} : (!cc.stdvec<!cc.measure_handle>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_INT:.*]] = call @__nvqpp_cudaqConvertToInteger(%{{.*}}) : (!cc.stdvec<i1>) -> i64
// CHECK:           return %[[VAL_INT]] : i64
// CHECK:         }

// 6. `mx_handle` / `my_handle` parity: single-qubit forms take the same path
//    through the bridge.
struct MxMyHandles {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle hx = mx_handle(q);
    cudaq::measure_handle hy = my_handle(q);
    (void)hx;
    (void)hy;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__MxMyHandles()
// CHECK:           quake.mx %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           quake.my %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           return
// CHECK:         }
