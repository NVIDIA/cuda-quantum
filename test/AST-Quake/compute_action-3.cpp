/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %cpp_std %s | cudaq-opt --lambda-lifting --canonicalize --apply-op-specialization | FileCheck %s
// clang-format on

#include <cudaq.h>

struct not_entry {
  auto operator()() __qpu__ {
    cudaq::qubit p;
    cudaq::qubit q;
    cudaq::compute_action([&]() {},
                          [&]() {
                            for (size_t i = 0; i < 1; ++i) {
                            }
                          });
  }
};

struct entry {
  auto operator()() __qpu__ {
    cudaq::qubit p;
    cudaq::qubit q;
    cudaq::compute_action([&]() { x(p); },
                          [&]() {
                            for (size_t i = 0; i < 1; ++i) {
                              y(q);
                            }
                          });
  }
};

int not_main() {
  auto counts = cudaq::sample(not_entry{});
  counts.dump();
  return 0;
}

int main() {
  auto counts = cudaq::sample(entry{});
  counts.dump();
  return 0;
}

// clang-format off
// CHECK-DAG:   func.func private @__nvqpp__lifted.lambda.0.adj() {
// CHECK-DAG:   func.func private @__nvqpp__lifted.lambda.2.adj(

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__not_entry()
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           call @__nvqpp__lifted.lambda.0() : () -> ()
// CHECK:           call @__nvqpp__lifted.lambda.1(%[[VAL_0]], %[[VAL_1]]) : (i64, i64) -> ()
// CHECK:           call @__nvqpp__lifted.lambda.0.adj() : () -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__entry()
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           call @__nvqpp__lifted.lambda.2(%[[VAL_2]]) : (!quake.ref) -> ()
// CHECK:           call @__nvqpp__lifted.lambda.3(%[[VAL_0]], %[[VAL_1]], %[[VAL_3]]) : (i64, i64, !quake.ref) -> ()
// CHECK:           call @__nvqpp__lifted.lambda.2.adj(%[[VAL_2]]) : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-DAG:   func.func private @__nvqpp__callable.thunk.lambda.2(
// CHECK-DAG:   func.func private @__nvqpp__lifted.lambda.2(
// CHECK-DAG:   func.func private @__nvqpp__callable.thunk.lambda.1(
// CHECK-DAG:   func.func private @__nvqpp__lifted.lambda.1(
// CHECK-DAG:   func.func private @__nvqpp__callable.thunk.lambda.0(
// CHECK-DAG:   func.func private @__nvqpp__lifted.lambda.0() {
// CHECK-DAG:   func.func private @__nvqpp__callable.thunk.lambda.3(
// CHECK-DAG:   func.func private @__nvqpp__lifted.lambda.3(
