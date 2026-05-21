/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --cse | FileCheck %s

#include <cudaq.h>

// These "bell" tests are very similar. Each tests a slightly different syntax
// for the equality test of the bits being measured.

struct bell {
  void operator()(int num_iters) __qpu__ {
    cudaq::qvector q(2);
    int n = 0;
    for (int i = 0; i < num_iters; i++) {
      h(q[0]);
      x<cudaq::ctrl>(q[0], q[1]);
      auto results = mz(q);
      bool r0 = results[0];
      if (r0 == results[1]) {
        n++;
      }
    }
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__bell(
// CHECK-SAME:      %[[ARG0:.*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[ALLOCA_0:.*]] = cc.alloca i32
// CHECK:           cc.store %[[ARG0]], %[[ALLOCA_0]] : !cc.ptr<i32>
// CHECK:           %[[ALLOCA_1:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[ALLOCA_2:.*]] = cc.alloca i32
// CHECK:           cc.store %[[CONSTANT_1]], %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:           cc.scope {
// CHECK:             %[[ALLOCA_3:.*]] = cc.alloca i32
// CHECK:             cc.store %[[CONSTANT_1]], %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[LOAD_0:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:               %[[LOAD_1:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i32>
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[LOAD_0]], %[[LOAD_1]] : i32
// CHECK:               cc.condition %[[CMPI_0]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_1]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:                 quake.h %[[EXTRACT_REF_0]] : (!quake.ref) -> ()
// CHECK:                 %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ALLOCA_1]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:                 quake.x {{\[}}%[[EXTRACT_REF_0]]] %[[EXTRACT_REF_1]] : (!quake.ref, !quake.ref) -> ()
// CHECK:                 %[[MZ_0:.*]] = quake.mz %[[ALLOCA_1]] name "results" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:                 %[[ALLOCA_4:.*]] = cc.alloca !cc.stdvec<!cc.measure_handle>
// CHECK:                 cc.store %[[MZ_0]], %[[ALLOCA_4]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:                 %[[LOAD_2:.*]] = cc.load %[[ALLOCA_4]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:                 %[[STDVEC_DATA_0:.*]] = cc.stdvec_data %[[LOAD_2]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.ptr<!cc.array<!cc.measure_handle x ?>>
// CHECK:                 %[[CAST_0:.*]] = cc.cast %[[STDVEC_DATA_0]] : (!cc.ptr<!cc.array<!cc.measure_handle x ?>>) -> !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[LOAD_3:.*]] = cc.load %[[CAST_0]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[LOAD_3]] : (!cc.measure_handle) -> i1
// CHECK:                 %[[ALLOCA_5:.*]] = cc.alloca i1
// CHECK:                 cc.store %[[DISCRIMINATE_0]], %[[ALLOCA_5]] : !cc.ptr<i1>
// CHECK:                 %[[LOAD_4:.*]] = cc.load %[[ALLOCA_5]] : !cc.ptr<i1>
// CHECK:                 %[[LOAD_5:.*]] = cc.load %[[ALLOCA_4]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:                 %[[STDVEC_DATA_1:.*]] = cc.stdvec_data %[[LOAD_5]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.ptr<!cc.array<!cc.measure_handle x ?>>
// CHECK:                 %[[COMPUTE_PTR_0:.*]] = cc.compute_ptr %[[STDVEC_DATA_1]][1] : (!cc.ptr<!cc.array<!cc.measure_handle x ?>>) -> !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[LOAD_6:.*]] = cc.load %[[COMPUTE_PTR_0]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[DISCRIMINATE_1:.*]] = quake.discriminate %[[LOAD_6]] : (!cc.measure_handle) -> i1
// CHECK:                 %[[CMPI_1:.*]] = arith.cmpi eq, %[[LOAD_4]], %[[DISCRIMINATE_1]] : i1
// CHECK:                 cc.if(%[[CMPI_1]]) {
// CHECK:                   %[[LOAD_7:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                   %[[ADDI_0:.*]] = arith.addi %[[LOAD_7]], %[[CONSTANT_0]] : i32
// CHECK:                   cc.store %[[ADDI_0]], %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[LOAD_8:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[LOAD_8]], %[[CONSTANT_0]] : i32
// CHECK:               cc.store %[[ADDI_1]], %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

struct libertybell {
  void operator()(int num_iters) __qpu__ {
    cudaq::qvector q(2);
    int n = 0;
    for (int i = 0; i < num_iters; i++) {
      h(q[0]);
      x<cudaq::ctrl>(q[0], q[1]);
      auto results = mz(q);
      if (results[0] == results[1]) {
        n++;
      }
    }
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__libertybell(
// CHECK-SAME:      %[[ARG0:.*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[ALLOCA_0:.*]] = cc.alloca i32
// CHECK:           cc.store %[[ARG0]], %[[ALLOCA_0]] : !cc.ptr<i32>
// CHECK:           %[[ALLOCA_1:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[ALLOCA_2:.*]] = cc.alloca i32
// CHECK:           cc.store %[[CONSTANT_1]], %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:           cc.scope {
// CHECK:             %[[ALLOCA_3:.*]] = cc.alloca i32
// CHECK:             cc.store %[[CONSTANT_1]], %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[LOAD_0:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:               %[[LOAD_1:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i32>
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[LOAD_0]], %[[LOAD_1]] : i32
// CHECK:               cc.condition %[[CMPI_0]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_1]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:                 quake.h %[[EXTRACT_REF_0]] : (!quake.ref) -> ()
// CHECK:                 %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ALLOCA_1]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:                 quake.x {{\[}}%[[EXTRACT_REF_0]]] %[[EXTRACT_REF_1]] : (!quake.ref, !quake.ref) -> ()
// CHECK:                 %[[MZ_0:.*]] = quake.mz %[[ALLOCA_1]] name "results" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:                 %[[ALLOCA_4:.*]] = cc.alloca !cc.stdvec<!cc.measure_handle>
// CHECK:                 cc.store %[[MZ_0]], %[[ALLOCA_4]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:                 %[[LOAD_2:.*]] = cc.load %[[ALLOCA_4]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:                 %[[STDVEC_DATA_0:.*]] = cc.stdvec_data %[[LOAD_2]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.ptr<!cc.array<!cc.measure_handle x ?>>
// CHECK:                 %[[CAST_0:.*]] = cc.cast %[[STDVEC_DATA_0]] : (!cc.ptr<!cc.array<!cc.measure_handle x ?>>) -> !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[LOAD_3:.*]] = cc.load %[[CAST_0]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[LOAD_3]] : (!cc.measure_handle) -> i1
// CHECK:                 %[[COMPUTE_PTR_0:.*]] = cc.compute_ptr %[[STDVEC_DATA_0]][1] : (!cc.ptr<!cc.array<!cc.measure_handle x ?>>) -> !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[LOAD_4:.*]] = cc.load %[[COMPUTE_PTR_0]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[DISCRIMINATE_1:.*]] = quake.discriminate %[[LOAD_4]] : (!cc.measure_handle) -> i1
// CHECK:                 %[[CMPI_1:.*]] = arith.cmpi eq, %[[DISCRIMINATE_0]], %[[DISCRIMINATE_1]] : i1
// CHECK:                 cc.if(%[[CMPI_1]]) {
// CHECK:                   %[[LOAD_5:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                   %[[ADDI_0:.*]] = arith.addi %[[LOAD_5]], %[[CONSTANT_0]] : i32
// CHECK:                   cc.store %[[ADDI_0]], %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[LOAD_6:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[LOAD_6]], %[[CONSTANT_0]] : i32
// CHECK:               cc.store %[[ADDI_1]], %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

struct tinkerbell {
  void operator()(int num_iters) __qpu__ {
    cudaq::qvector q(2);
    int n = 0;
    for (int i = 0; i < num_iters; i++) {
      h(q[0]);
      x<cudaq::ctrl>(q[0], q[1]);
      auto results = mz(q);
      auto r0 = results[0];
      auto r1 = results[1];
      if (r0 == r1) {
        n++;
      }
    }
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__tinkerbell(
// CHECK-SAME:      %[[ARG0:.*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[ALLOCA_0:.*]] = cc.alloca i32
// CHECK:           cc.store %[[ARG0]], %[[ALLOCA_0]] : !cc.ptr<i32>
// CHECK:           %[[ALLOCA_1:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[ALLOCA_2:.*]] = cc.alloca i32
// CHECK:           cc.store %[[CONSTANT_1]], %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:           cc.scope {
// CHECK:             %[[ALLOCA_3:.*]] = cc.alloca i32
// CHECK:             cc.store %[[CONSTANT_1]], %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[LOAD_0:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:               %[[LOAD_1:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i32>
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[LOAD_0]], %[[LOAD_1]] : i32
// CHECK:               cc.condition %[[CMPI_0]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_1]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:                 quake.h %[[EXTRACT_REF_0]] : (!quake.ref) -> ()
// CHECK:                 %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ALLOCA_1]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:                 quake.x {{\[}}%[[EXTRACT_REF_0]]] %[[EXTRACT_REF_1]] : (!quake.ref, !quake.ref) -> ()
// CHECK:                 %[[MZ_0:.*]] = quake.mz %[[ALLOCA_1]] name "results" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:                 %[[ALLOCA_4:.*]] = cc.alloca !cc.stdvec<!cc.measure_handle>
// CHECK:                 cc.store %[[MZ_0]], %[[ALLOCA_4]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:                 %[[LOAD_2:.*]] = cc.load %[[ALLOCA_4]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:                 %[[STDVEC_DATA_0:.*]] = cc.stdvec_data %[[LOAD_2]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.ptr<!cc.array<!cc.measure_handle x ?>>
// CHECK:                 %[[CAST_0:.*]] = cc.cast %[[STDVEC_DATA_0]] : (!cc.ptr<!cc.array<!cc.measure_handle x ?>>) -> !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[LOAD_3:.*]] = cc.load %[[CAST_0]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[ALLOCA_5:.*]] = cc.alloca !cc.measure_handle
// CHECK:                 cc.store %[[LOAD_3]], %[[ALLOCA_5]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[LOAD_4:.*]] = cc.load %[[ALLOCA_4]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:                 %[[STDVEC_DATA_1:.*]] = cc.stdvec_data %[[LOAD_4]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.ptr<!cc.array<!cc.measure_handle x ?>>
// CHECK:                 %[[COMPUTE_PTR_0:.*]] = cc.compute_ptr %[[STDVEC_DATA_1]][1] : (!cc.ptr<!cc.array<!cc.measure_handle x ?>>) -> !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[LOAD_5:.*]] = cc.load %[[COMPUTE_PTR_0]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[ALLOCA_6:.*]] = cc.alloca !cc.measure_handle
// CHECK:                 cc.store %[[LOAD_5]], %[[ALLOCA_6]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[LOAD_6:.*]] = cc.load %[[ALLOCA_5]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[LOAD_6]] : (!cc.measure_handle) -> i1
// CHECK:                 %[[LOAD_7:.*]] = cc.load %[[ALLOCA_6]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[DISCRIMINATE_1:.*]] = quake.discriminate %[[LOAD_7]] : (!cc.measure_handle) -> i1
// CHECK:                 %[[CMPI_1:.*]] = arith.cmpi eq, %[[DISCRIMINATE_0]], %[[DISCRIMINATE_1]] : i1
// CHECK:                 cc.if(%[[CMPI_1]]) {
// CHECK:                   %[[LOAD_8:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                   %[[ADDI_0:.*]] = arith.addi %[[LOAD_8]], %[[CONSTANT_0]] : i32
// CHECK:                   cc.store %[[ADDI_0]], %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[LOAD_9:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[LOAD_9]], %[[CONSTANT_0]] : i32
// CHECK:               cc.store %[[ADDI_1]], %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on
