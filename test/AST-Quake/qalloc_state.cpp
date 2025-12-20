/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct Eins {
  std::vector<bool> operator()(cudaq::state *state) __qpu__ {
    cudaq::qvector v(state);
    h(v);
    return mz(v);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Eins(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.ptr<!quake.state>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_3:.*]] = quake.get_number_of_qubits %[[VAL_0]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>[%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_0]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>

struct Zwei {
  std::vector<bool> operator()(const cudaq::state *state) __qpu__ {
    cudaq::qvector v(state);
    h(v);
    return mz(v);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Zwei(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.ptr<!quake.state>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_3:.*]] = quake.get_number_of_qubits %[[VAL_0]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>[%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_0]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>

struct Drei {
  std::vector<bool> operator()(cudaq::state &state) __qpu__ {
    cudaq::qvector v(state);
    h(v);
    return mz(v);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Drei(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.ptr<!quake.state>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_3:.*]] = quake.get_number_of_qubits %[[VAL_0]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>[%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_0]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>

struct Vier {
  std::vector<bool> operator()(const cudaq::state &state) __qpu__ {
    cudaq::qvector v(state);
    h(v);
    return mz(v);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Vier(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.ptr<!quake.state>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_3:.*]] = quake.get_number_of_qubits %[[VAL_0]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>[%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_0]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>

