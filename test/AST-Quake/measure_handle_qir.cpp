/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --expand-measurements --canonicalize --lower-cc-measure-handle --convert-to-qir-api --symbol-dce | FileCheck %s
// clang-format on

// End-to-end regression for the single-API `measure_handle` surface across
// the full lowering pipeline. Two kernels exercise the two host-device
// boundary shapes the spec allows:
//
//   * `bool` return from `mz(qubit)` -- the bridge stages the handle
//     through `cc.alloca`/`cc.store`/`cc.load` and discriminates at the
//     return site. After `lower-cc-measure-handle` the handle is i64;
//     after `convert-to-qir-api` the discriminate becomes a Result*->i1
//     read.
//
//   * `std::vector<bool>` return from `to_bools(mz(qvec))` -- the bridge
//     emits a vectorized `quake.discriminate`, `expand-measurements`
//     unrolls the qvec measurement into per-qubit `mz` calls, and the QIR
//     API emits the standard `__nvqpp_vectorCopyCtor` / `cc.stdvec_init`
//     return prologue.
//
// This test guards the pipeline ordering: in particular,
// `lower-cc-measure-handle` must run before `convert-to-qir-api` so the
// pointer-to-handle elementType is consistent when the QIR conversion
// rewrites measurement calls.

#include <cudaq.h>

struct ScalarReturn {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    return mz(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ScalarReturn() -> i1
// CHECK:           %[[VAL_Q:.*]] = call @__quantum__rt__qubit_allocate()
// CHECK:           call @__quantum__qis__h(%[[VAL_Q]])
// CHECK:           %[[VAL_R:.*]] = call @__quantum__qis__mz(%[[VAL_Q]]) {{.*}} -> !cc.ptr<!llvm.struct<"Result", opaque>>
// CHECK:           %[[VAL_I:.*]] = cc.cast %[[VAL_R]] : (!cc.ptr<!llvm.struct<"Result", opaque>>) -> i64
// CHECK:           %[[VAL_S:.*]] = cc.alloca i64
// CHECK:           cc.store %[[VAL_I]], %[[VAL_S]] : !cc.ptr<i64>
// CHECK:           %[[VAL_L:.*]] = cc.load %[[VAL_S]] : !cc.ptr<i64>
// CHECK:           %[[VAL_P:.*]] = cc.cast %[[VAL_L]] : (i64) -> !cc.ptr<i1>
// CHECK:           %[[VAL_B:.*]] = cc.load %[[VAL_P]] : !cc.ptr<i1>
// CHECK:           return %[[VAL_B]] : i1
// CHECK:         }

struct VectorReturn {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector v(3);
    h(v);
    return cudaq::to_bools(mz(v));
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorReturn() -> !cc.stdvec<i1>
// CHECK:           %[[V_C3:.*]] = arith.constant 3 : i64
// CHECK:           %[[V_ARR:.*]] = call @__quantum__rt__qubit_allocate_array(%[[V_C3]])
// CHECK:           cc.loop while
// CHECK:             %[[V_QP:.*]] = func.call @__quantum__rt__array_get_element_ptr_1d(%[[V_ARR]], %{{.*}})
// CHECK:             %[[V_Q:.*]] = cc.load %[[V_QP]]
// CHECK:             func.call @__quantum__qis__h(%[[V_Q]])
// CHECK:           %[[V_BUF:.*]] = cc.alloca !cc.array<i8 x 3>
// CHECK:           cc.loop while
// CHECK:             %[[V_QP2:.*]] = func.call @__quantum__rt__array_get_element_ptr_1d(%[[V_ARR]], %{{.*}})
// CHECK:             %[[V_Q2:.*]] = cc.load %[[V_QP2]]
// CHECK:             %[[V_R:.*]] = func.call @__quantum__qis__mz(%[[V_Q2]]) {{.*}} -> !cc.ptr<!llvm.struct<"Result", opaque>>
// CHECK:             %[[V_RP:.*]] = cc.cast %[[V_R]] : (!cc.ptr<!llvm.struct<"Result", opaque>>) -> !cc.ptr<i1>
// CHECK:             %[[V_B:.*]] = cc.load %[[V_RP]]
// CHECK:             %[[V_SP:.*]] = cc.compute_ptr %[[V_BUF]][%{{.*}}]
// CHECK:             %[[V_BB:.*]] = cc.cast unsigned %[[V_B]] : (i1) -> i8
// CHECK:             cc.store %[[V_BB]], %[[V_SP]]
// CHECK:           %[[V_BUFP:.*]] = cc.cast %[[V_BUF]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           %[[V_HEAP:.*]] = call @__nvqpp_vectorCopyCtor(%[[V_BUFP]], %[[V_C3]], %{{.*}})
// CHECK:           %[[V_VEC:.*]] = cc.stdvec_init %[[V_HEAP]], %[[V_C3]]
// CHECK:           return %[[V_VEC]] : !cc.stdvec<i1>
// CHECK:         }
