/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
// CHECK:           %[[VAL_4:.*]] = call @__nvqpp_cudaq_state_vectorData(%[[VAL_0]]) : (!cc.ptr<!cc.state>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_4]] : (!quake.veq<?>, !cc.ptr<f64>) -> !quake.veq<?>
// CHECK:           %[[VAL_7:.*]] = quake.veq_size %[[VAL_6]] : (!quake.veq<?>) -> i64

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
// CHECK:           %[[VAL_4:.*]] = call @__nvqpp_cudaq_state_vectorData(%[[VAL_0]]) : (!cc.ptr<!cc.state>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_4]] : (!quake.veq<?>, !cc.ptr<f64>) -> !quake.veq<?>
// CHECK:           %[[VAL_7:.*]] = quake.veq_size %[[VAL_6]] : (!quake.veq<?>) -> i64

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
// CHECK:           %[[VAL_4:.*]] = call @__nvqpp_cudaq_state_vectorData(%[[VAL_0]]) : (!cc.ptr<!cc.state>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_4]] : (!quake.veq<?>, !cc.ptr<f64>) -> !quake.veq<?>
// CHECK:           %[[VAL_7:.*]] = quake.veq_size %[[VAL_6]] : (!quake.veq<?>) -> i64

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
// CHECK:           %[[VAL_4:.*]] = call @__nvqpp_cudaq_state_vectorData(%[[VAL_0]]) : (!cc.ptr<!cc.state>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_4]] : (!quake.veq<?>, !cc.ptr<f64>) -> !quake.veq<?>
// CHECK:           %[[VAL_7:.*]] = quake.veq_size %[[VAL_6]] : (!quake.veq<?>) -> i64

#if 0
// rvalue reference isn't supported yet.
struct Fuenf {
  std::vector<bool> operator()(cudaq::state &&state) __qpu__ {
    cudaq::qvector v(std::move(state));
    h(v);
    return mz(v);
  }
};
#endif

// CHECK: func.func private @__nvqpp_cudaq_state_vectorData(!cc.ptr<!cc.state>) -> !cc.ptr<f64>
// CHECK: func.func private @__nvqpp_cudaq_state_numberOfQubits(!cc.ptr<!cc.state>) -> i64
