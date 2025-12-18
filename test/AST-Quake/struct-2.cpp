/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

// Check arguments of type struct<vector<T>...>.
struct ProductOfVector {
  std::vector<int> m1;
  std::vector<double> m2;
};

struct Qernel0 {
  void operator()(ProductOfVector arg) __qpu__ {
    [[maybe_unused]] int i{arg.m1[0]};
    // the most interesting function in the world goes here
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Qernel0(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"ProductOfVector" {!cc.stdvec<i32>, !cc.stdvec<f64>} [384,8]>)
// CHECK:           %[[VAL_1:.*]] = cc.alloca !cc.struct<"ProductOfVector" {!cc.stdvec<i32>, !cc.stdvec<f64>} [384,8]>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<!cc.struct<"ProductOfVector" {!cc.stdvec<i32>, !cc.stdvec<f64>} [384,8]>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.struct<"ProductOfVector" {!cc.stdvec<i32>, !cc.stdvec<f64>} [384,8]>>) -> !cc.ptr<!cc.stdvec<i32>>
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<!cc.stdvec<i32>>
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_data %[[VAL_3]] : (!cc.stdvec<i32>) -> !cc.ptr<!cc.array<i32 x ?>>
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<i32 x ?>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_6]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           return
// CHECK:         }

struct Pi0 {
  int m1;
  float m2;
};

struct Pi1 {
  char m1;
  double m2;
};

// Check arguments of type struct<struct<Ts...>...>.
struct ProductOfPis {
  Pi0 m_0;
  Pi1 m_1;
};

struct Qernel1 {
  void operator()(ProductOfPis arg) __qpu__ {
    [[maybe_unused]] char i{arg.m_1.m1};
    // the most flamboyant function in the world goes here
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Qernel1(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"ProductOfPis" {!cc.struct<"Pi0" {i32, f32} [64,4]>, !cc.struct<"Pi1" {i8, f64} [128,8]>} [192,8]>)
// CHECK:           %[[VAL_1:.*]] = cc.alloca !cc.struct<"ProductOfPis" {!cc.struct<"Pi0" {i32, f32} [64,4]>, !cc.struct<"Pi1" {i8, f64} [128,8]>} [192,8]>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<!cc.struct<"ProductOfPis" {!cc.struct<"Pi0" {i32, f32} [64,4]>, !cc.struct<"Pi1" {i8, f64} [128,8]>} [192,8]>>
// CHECK:           %[[VAL_2:.*]] = cc.compute_ptr %[[VAL_1]][1, 0] : (!cc.ptr<!cc.struct<"ProductOfPis" {!cc.struct<"Pi0" {i32, f32} [64,4]>, !cc.struct<"Pi1" {i8, f64} [128,8]>} [192,8]>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i8>
// CHECK:           %[[VAL_5:.*]] = cc.alloca i8
// CHECK:           cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<i8>
// CHECK:           return
// CHECK:         }

struct Product {
  short m1;
  float m2;
};

// Check arguments of type vector<struct<Ts...>>.
using VectorOfProduct = std::vector<Product>;

struct Qernel2 {
  void operator()(VectorOfProduct arg) __qpu__ {
    [[maybe_unused]] short i{arg[0].m1};
    // the most swashbuckling function in the world goes here
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Qernel2(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<!cc.struct<"Product" {i16, f32} [64,4]>>)
// CHECK:           %[[VAL_1:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<!cc.struct<"Product" {i16, f32} [64,4]>>) -> !cc.ptr<!cc.array<!cc.struct<"Product" {i16, f32} [64,4]> x ?>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<!cc.struct<"Product" {i16, f32} [64,4]> x ?>>) -> !cc.ptr<i16>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i16>
// CHECK:           %[[VAL_5:.*]] = cc.alloca i16
// CHECK:           cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<i16>
// CHECK:           return
// CHECK:         }

