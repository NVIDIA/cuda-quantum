/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__range_for_measure_auto()
// CHECK:           quake.mz %{{.*}} : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           cc.store %{{.*}}, %{{.*}} : !cc.ptr<!cc.measure_handle>
