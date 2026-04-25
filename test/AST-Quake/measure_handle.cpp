/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// Exercise the C++ AST bridge for the single-API `cudaq::measure_handle`
// surface defined by the `measure_handle` proposal:
//
//   * `cudaq::mz/mx/my(qubit&)`  => `quake.mz/mx/my ... -> !cc.measure_handle`
//   * `cudaq::mz/mx/my(qvec)`    => `quake.mz/mx/my ... -> !cc.stdvec<!cc.measure_handle>`
//   * `cudaq::to_bools(vec<h>)`  => `quake.discriminate ... -> !cc.stdvec<i1>`
//   * `measure_handle::operator bool()` (non-explicit)
//                                => `cc.alloca/cc.store/cc.load/quake.discriminate`
//
// Importantly, the handle-returning measurement overloads must NOT inline
// a `quake.discriminate`; only contexts that demand a `bool` (assignment
// to `bool`, `if`/`while` condition, `==`/`&&`/`||`, ternary condition,
// bool return) trigger the discriminate.

#include <cudaq.h>

// 1. Scalar handle bound to a `cudaq::measure_handle` and never coerced.
//    The bridge stores the handle into an `!cc.measure_handle` alloca but
//    emits no `quake.discriminate`.
struct ScalarHandle {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle h = mz(q);
    (void)h;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ScalarHandle()
// CHECK:           %[[VAL_Q:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_Q]] name "h" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M]], %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK-NOT:       quake.discriminate
// CHECK:           return
// CHECK:         }

// 2. Implicit bool coercion through `return h;`. The bridge inserts the
//    alloca/store/load round-trip and a single `quake.discriminate`.
struct ScalarBool {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    auto h = mz(q);
    return h;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ScalarBool() -> i1
// CHECK:           %[[VAL_Q:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_Q]] name "h" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M]], %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_HL:.*]] = cc.load %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_B:.*]] = quake.discriminate %[[VAL_HL]] : (!cc.measure_handle) -> i1
// CHECK:           return %[[VAL_B]] : i1
// CHECK:         }

// 3. Direct expression coercion in an `if` condition. Same alloca/store/
//    load/discriminate splice; no intermediate handle binding.
struct DirectIf {
  void operator()() __qpu__ {
    cudaq::qubit q;
    if (mz(q))
      x(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__DirectIf()
// CHECK:           %[[VAL_Q:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_Q]] : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M]], %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_HL:.*]] = cc.load %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_B:.*]] = quake.discriminate %[[VAL_HL]] : (!cc.measure_handle) -> i1
// CHECK:           cc.if(%[[VAL_B]]) {
// CHECK:             quake.x %[[VAL_Q]]
// CHECK:           }
// CHECK:           return
// CHECK:         }

// 4. Range-form `mz(qvec)` produces a stdvec of handles with no inlined
//    discriminate.
struct RegisterHandle {
  void operator()() __qpu__ {
    cudaq::qvector v(4);
    auto hs = mz(v);
    (void)hs;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__RegisterHandle()
// CHECK:           %[[VAL_V:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %{{.*}} = quake.mz %[[VAL_V]] name "hs" : (!quake.veq<4>) -> !cc.stdvec<!cc.measure_handle>
// CHECK-NOT:       quake.discriminate
// CHECK:           return
// CHECK:         }

// 5. Bulk discrimination through `cudaq::to_bools` lowers to a vectorized
//    `quake.discriminate` consuming `!cc.stdvec<!cc.measure_handle>` and
//    producing `!cc.stdvec<i1>`. The bridge then emits the standard vector
//    return prologue (data/size/copyCtor/init) so the caller receives a
//    `std::vector<bool>`.
struct RegisterBools {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector v(4);
    return cudaq::to_bools(mz(v));
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__RegisterBools() -> !cc.stdvec<i1>
// CHECK:           %[[VAL_V:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_V]] : (!quake.veq<4>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[VAL_BV:.*]] = quake.discriminate %[[VAL_M]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.stdvec<i1>
// CHECK:           %{{.*}} = cc.stdvec_data %[[VAL_BV]] : (!cc.stdvec<i1>) -> !cc.ptr<i8>
// CHECK:           %{{.*}} = cc.stdvec_size %[[VAL_BV]] : (!cc.stdvec<i1>) -> i64
// CHECK:           %{{.*}} = call @__nvqpp_vectorCopyCtor(
// CHECK:           return %{{.*}} : !cc.stdvec<i1>
// CHECK:         }

// 6. `mx` and `my` parity: single-qubit forms take the same bridge path
//    as `mz` and produce `!cc.measure_handle` SSA values.
struct MxMyHandles {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle hx = mx(q);
    cudaq::measure_handle hy = my(q);
    (void)hx;
    (void)hy;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__MxMyHandles()
// CHECK:           quake.mx %{{.*}} name "hx" : (!quake.ref) -> !cc.measure_handle
// CHECK:           quake.my %{{.*}} name "hy" : (!quake.ref) -> !cc.measure_handle
// CHECK-NOT:       quake.discriminate
// CHECK:           return
// CHECK:         }

// 6b. `mx` and `my` range parity: the qvec forms must take the same
//     bridge path as `mz(qvec)` and produce
//     `!cc.stdvec<!cc.measure_handle>` -- without instantiating the
//     library-mode C++ template bodies in `qubit_qis.h` (the bridge
//     intercepts the call by name in `ConvertExpr.cpp`).
struct MxMyRange {
  void operator()() __qpu__ {
    cudaq::qvector qv(3);
    auto hx = mx(qv);
    auto hy = my(qv);
    (void)hx;
    (void)hy;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__MxMyRange()
// CHECK:           quake.mx %{{.*}} name "hx" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           quake.my %{{.*}} name "hy" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
// CHECK-NOT:       quake.discriminate
// CHECK:           return
// CHECK:         }

// 7. Cross-function passage. A pure-device `__qpu__` function takes a
//    qubit reference and a `const cudaq::measure_handle &`; the qubit
//    keeps the callee out of entry-point classification, so the boundary
//    rule does not forbid the handle parameter. The handle reaches the
//    callee through a `!cc.ptr<!cc.measure_handle>`, the callee loads it
//    once, and the bool coercion in the body lowers to `quake.discriminate`.
__qpu__ bool consume_handle(cudaq::qubit &q, const cudaq::measure_handle &h) {
  return h;
}

struct CrossFunctionCaller {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    auto h = mz(q);
    return consume_handle(q, h);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_consume_handle.
// CHECK-SAME:      (%{{.*}}: !quake.ref{{.*}}, %[[ARG_H:.*]]: !cc.ptr<!cc.measure_handle>{{.*}}) -> i1
// CHECK:           %[[VAL_HL:.*]] = cc.load %[[ARG_H]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_B:.*]] = quake.discriminate %[[VAL_HL]] : (!cc.measure_handle) -> i1
// CHECK:           return %[[VAL_B]] : i1
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CrossFunctionCaller() -> i1
// CHECK:           %[[VAL_Q:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_Q]] name "h" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M]], %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_R:.*]] = call @__nvqpp__mlirgen__function_consume_handle.{{.*}}(%[[VAL_Q]], %[[VAL_HA]]) : (!quake.ref, !cc.ptr<!cc.measure_handle>) -> i1
// CHECK:           return %[[VAL_R]] : i1
// CHECK:         }

// 8. Equality between two handles flows through the implicit `bool`
//    conversion, NOT through any handle-level `operator==`. Each operand
//    discriminates independently; the comparison runs on the resulting
//    `i1`s, so `h1 == h2` denotes outcome equality, not handle identity.
struct HandleEquality {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::qubit r;
    auto h1 = mz(q);
    auto h2 = mz(r);
    return h1 == h2;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__HandleEquality() -> i1
// CHECK:           %[[VAL_M1:.*]] = quake.mz %{{.*}} name "h1" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA1:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M1]], %[[VAL_HA1]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_M2:.*]] = quake.mz %{{.*}} name "h2" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA2:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M2]], %[[VAL_HA2]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_HL1:.*]] = cc.load %[[VAL_HA1]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_B1:.*]] = quake.discriminate %[[VAL_HL1]] : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_HL2:.*]] = cc.load %[[VAL_HA2]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_B2:.*]] = quake.discriminate %[[VAL_HL2]] : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_EQ:.*]] = arith.cmpi eq, %[[VAL_B1]], %[[VAL_B2]] : i1
// CHECK:           return %[[VAL_EQ]] : i1
// CHECK:         }

// 9. Short-circuit `&&` over two coerced handles. The first handle is
//    discriminated unconditionally; the second `mz` and its discriminate
//    live inside the else branch of the short-circuit `cc.if`.
struct HandleAnd {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::qubit r;
    return mz(q) && mz(r);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__HandleAnd() -> i1
// CHECK:           %[[VAL_FALSE:.*]] = arith.constant false
// CHECK:           %[[VAL_M1:.*]] = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA1:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M1]], %[[VAL_HA1]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_HL1:.*]] = cc.load %[[VAL_HA1]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_B1:.*]] = quake.discriminate %[[VAL_HL1]] : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_NOT:.*]] = arith.cmpi eq, %[[VAL_B1]], %[[VAL_FALSE]] : i1
// CHECK:           %[[VAL_RES:.*]] = cc.if(%[[VAL_NOT]]) -> i1 {
// CHECK:             cc.continue %[[VAL_FALSE]] : i1
// CHECK:           } else {
// CHECK:             %[[VAL_M2:.*]] = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:             %[[VAL_HA2:.*]] = cc.alloca !cc.measure_handle
// CHECK:             cc.store %[[VAL_M2]], %[[VAL_HA2]] : !cc.ptr<!cc.measure_handle>
// CHECK:             %[[VAL_HL2:.*]] = cc.load %[[VAL_HA2]] : !cc.ptr<!cc.measure_handle>
// CHECK:             %[[VAL_B2:.*]] = quake.discriminate %[[VAL_HL2]] : (!cc.measure_handle) -> i1
// CHECK:             cc.continue %[[VAL_B2]] : i1
// CHECK:           }
// CHECK:           return %[[VAL_RES]] : i1
// CHECK:         }
