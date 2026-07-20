/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>
#include <vector>

// A range-based for loop with a `bool` loop variable over the result of a
// multi-qubit `mz` iterates a container of `!cc.measure_handle`. The bridge
// must lower each per-iteration handle through `quake.discriminate` before
// storing it into the `i1` slot of the loop variable. Without it the bridge
// emitted `cc.store %handle, %i1ptr`, whose value/pointer types disagree, and
// the verifier crashed `cudaq-quake`.

struct range_for_measure_bool {
  bool operator()() __qpu__ {
    cudaq::qarray<2> reg;
    bool any = false;
    for (bool b : mz(reg)) {
      any = any || b;
    }
    return any;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__range_for_measure_bool()
// CHECK:           quake.mz %{{.*}} : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[HANDLE:.*]] = cc.load %{{.*}} : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[BIT:.*]] = quake.discriminate %[[HANDLE]] : (!cc.measure_handle) -> i1
// CHECK:           cc.store %[[BIT]], %{{.*}} : !cc.ptr<i1>
// clang-format on

// A `measure_handle` loop variable (`auto`) keeps the handle and must NOT have
// a discriminate inserted at bind time; the discriminate happens at the
// `operator bool()` use site instead.

struct range_for_measure_auto {
  bool operator()() __qpu__ {
    cudaq::qarray<2> reg;
    bool any = false;
    for (auto b : mz(reg)) {
      any = any || b;
    }
    return any;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__range_for_measure_auto()
// CHECK:           quake.mz %{{.*}} : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           cc.store %{{.*}}, %{{.*}} : !cc.ptr<!cc.measure_handle>
// clang-format on

// A named `std::vector<measure_handle>` bound straight from a multi-qubit `mz`.
// The descriptor slot stores the `mz` result, so the vector is bound and the
// per-iteration `quake.discriminate` is still emitted for the `bool` loop
// variable (regression guard against over-diagnosing a bound vector).

struct range_for_measure_named_vector {
  bool operator()() __qpu__ {
    cudaq::qarray<2> reg;
    auto handles = mz(reg);
    bool any = false;
    for (bool b : handles) {
      any = any || b;
    }
    return any;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__range_for_measure_named_vector()
// CHECK:           quake.mz %{{.*}} name "handles" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           cc.store %{{.*}}, %{{.*}} : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           %[[HANDLE:.*]] = cc.load %{{.*}} : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[BIT:.*]] = quake.discriminate %[[HANDLE]] : (!cc.measure_handle) -> i1
// CHECK:           cc.store %[[BIT]], %{{.*}} : !cc.ptr<i1>
// clang-format on

// A list-initialized `std::vector<measure_handle>` whose backing array receives
// the individual `mz` results one hop away -- element 0 through a decay
// `cc.cast`, element 1 through a `cc.compute_ptr`. The vector is bound, so the
// loop discriminates rather than diagnosing (regression guard for the
// cast-reached backing-store path).

struct range_for_measure_list_init {
  bool operator()() __qpu__ {
    cudaq::qubit q0, q1;
    std::vector<cudaq::measure_handle> handles{mz(q0), mz(q1)};
    bool any = false;
    for (bool b : handles) {
      any = any || b;
    }
    return any;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__range_for_measure_list_init()
// CHECK:           quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           cc.stdvec_init %{{.*}}, %{{.*}} : (!cc.ptr<!cc.array<!cc.measure_handle x ?>>, i64) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[HANDLE:.*]] = cc.load %{{.*}} : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[BIT:.*]] = quake.discriminate %[[HANDLE]] : (!cc.measure_handle) -> i1
// CHECK:           cc.store %[[BIT]], %{{.*}} : !cc.ptr<i1>
// clang-format on
