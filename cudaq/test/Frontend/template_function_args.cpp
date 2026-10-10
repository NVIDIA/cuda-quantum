/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// Kernel templates whose arguments are not all types: non-type arguments
// (`bool`, `int`), mixtures, and parameter packs. Each specialization must get
// its own kernel.

#include <cudaq.h>

template <typename T>
__qpu__ int byType(T t) {
  if constexpr (sizeof(T) > 2)
    return 1;
  else
    return 2;
}

template <bool B>
__qpu__ int byBool() {
  if constexpr (B)
    return 1;
  return 0;
}

template <int N>
__qpu__ int byInt() {
  return N;
}

template <typename T, int N>
__qpu__ int mixed(T t) {
  return N + sizeof(T);
}

template <typename... Ts>
__qpu__ int pack(Ts... ts) {
  return sizeof...(Ts);
}

__qpu__ void use() {
  byType(1);
  byType('c');
  byBool<true>();
  byBool<false>();
  byInt<3>();
  byInt<4>();
  mixed<int, 5>(1);
  pack(1, 'c', 2.0, 3u, 4L, 5.0f);
}

// CHECK-DAG: func.func @__nvqpp__mlirgen__instance_function_byTypei.{{.*}}
// CHECK-DAG: func.func @__nvqpp__mlirgen__instance_function_byTypec.{{.*}}
// CHECK-DAG: func.func @__nvqpp__mlirgen__instance_function_byBool.{{.*}}Lb1E{{.*}}
// CHECK-DAG: func.func @__nvqpp__mlirgen__instance_function_byBool.{{.*}}Lb0E{{.*}}
// CHECK-DAG: func.func @__nvqpp__mlirgen__instance_function_byInt.{{.*}}Li3E{{.*}}
// CHECK-DAG: func.func @__nvqpp__mlirgen__instance_function_byInt.{{.*}}Li4E{{.*}}
// CHECK-DAG: func.func @__nvqpp__mlirgen__instance_function_mixedi.{{.*}}Li5E{{.*}}
// CHECK-DAG: func.func @__nvqpp__mlirgen__instance_function_pack{{.*}}
// The length of the pack is a constant in the specialization (six types).
// CHECK-DAG: arith.constant 6 : i32

// Each element of a pack is a distinct parameter.
template <typename... Ts>
__qpu__ int foldSum(Ts... ts) {
  return (ts + ...);
}
__qpu__ int useFold() { return foldSum(1, 2, 3); }

// CHECK-LABEL: func.func @__nvqpp__mlirgen__instance_function_foldSumiii.
// CHECK-SAME:    (%[[A0:.*]]: i32{{.*}}, %[[A1:.*]]: i32{{.*}}, %[[A2:.*]]: i32{{.*}})
// CHECK:         %[[S0:.*]] = cc.alloca i32
// CHECK:         cc.store %[[A0]], %[[S0]]
// CHECK:         %[[S1:.*]] = cc.alloca i32
// CHECK:         cc.store %[[A1]], %[[S1]]
// CHECK:         %[[S2:.*]] = cc.alloca i32
// CHECK:         cc.store %[[A2]], %[[S2]]
// CHECK:         cc.load %[[S0]]
// CHECK:         cc.load %[[S1]]
// CHECK:         cc.load %[[S2]]
