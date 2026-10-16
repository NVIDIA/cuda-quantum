/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include "cudaq.h"

// A type can be declared in a kernel, as it can be anywhere else. A declaration
// of a type generates no code. The type is converted where a variable uses it.
// (This used to be an error, "Cannot find a in the symbol table.")

struct LocalStruct {
  double operator()(double x) __qpu__ {
    struct Pair {
      int a;
      double b;
    };
    using Index = int;
    typedef double Scalar;
    Pair p{4, 2.5};
    Index i = 3;
    Scalar s = x;
    p.a = p.a + i;
    p.b = p.b * s;
    return p.b + p.a;
  }
};

// The declaration of a type can be a part of the declaration of a variable, and
// the type does not need to have a name.
struct DeclaredWithVariable {
  int operator()() __qpu__ {
    struct Named {
      int n;
    } named;
    struct {
      int m;
    } unnamed;
    named.n = 1;
    unnamed.m = 2;
    return named.n + unnamed.m;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__LocalStruct(
// CHECK-SAME:      %[[ARG:.*]]: f64{{.*}}) -> f64
// CHECK-DAG:       %[[P:.*]] = cc.alloca !cc.struct<"Pair" {i32, f64}
// CHECK-DAG:       %[[I:.*]] = cc.alloca i32
// CHECK-DAG:       %[[S:.*]] = cc.alloca f64
// The members of the struct are updated in place.
// CHECK:           %[[A:.*]] = cc.cast %[[P]] : (!cc.ptr<!cc.struct<"Pair" {i32, f64}{{.*}}>>) -> !cc.ptr<i32>
// CHECK:           arith.addi
// CHECK:           cc.store %{{.*}}, %[[A]] : !cc.ptr<i32>
// CHECK:           %[[B:.*]] = cc.compute_ptr %[[P]][1]
// CHECK:           arith.mulf
// CHECK:           cc.store %{{.*}}, %[[B]] : !cc.ptr<f64>
// CHECK:           arith.addf
// CHECK:           return %{{.*}} : f64

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__DeclaredWithVariable(
// CHECK-SAME:      ) -> i32
// CHECK-DAG:       cc.alloca !cc.struct<"Named" {i32}
// CHECK-DAG:       cc.alloca !cc.struct<{i32}
// CHECK:           return %{{.*}} : i32
// clang-format on
