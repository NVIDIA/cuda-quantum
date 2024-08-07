/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct Eins {
  std::vector<bool> operator()(cudaq::state *state) __qpu__ {
    cudaq::qvector v(state);
    h(v);
    return mz(v);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Eins(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.ptr<!cc.state>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_3:.*]] = call @__nvqpp_cudaq_state_numberOfQubits(%[[VAL_0]]) : (!cc.ptr<!cc.state>) -> i64
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>[%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_0]] : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>

struct Zwei {
  std::vector<bool> operator()(const cudaq::state *state) __qpu__ {
    cudaq::qvector v(state);
    h(v);
    return mz(v);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Zwei(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.ptr<!cc.state>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_3:.*]] = call @__nvqpp_cudaq_state_numberOfQubits(%[[VAL_0]]) : (!cc.ptr<!cc.state>) -> i64
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>[%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_0]] : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>

struct Drei {
  std::vector<bool> operator()(cudaq::state &state) __qpu__ {
    cudaq::qvector v(state);
    h(v);
    return mz(v);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Drei(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.ptr<!cc.state>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_3:.*]] = call @__nvqpp_cudaq_state_numberOfQubits(%[[VAL_0]]) : (!cc.ptr<!cc.state>) -> i64
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>[%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_0]] : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>

struct Vier {
  std::vector<bool> operator()(const cudaq::state &state) __qpu__ {
    cudaq::qvector v(state);
    h(v);
    return mz(v);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Vier(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.ptr<!cc.state>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_3:.*]] = call @__nvqpp_cudaq_state_numberOfQubits(%[[VAL_0]]) : (!cc.ptr<!cc.state>) -> i64
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>[%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_0]] : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>

// CHECK: func.func private @__nvqpp_cudaq_state_numberOfQubits(!cc.ptr<!cc.state>) -> i64
