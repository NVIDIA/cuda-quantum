/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s
// clang-format off

#include "cudaq.h"

// An enumeration is represented by its underlying integer type, and an
// enumerator is a constant of that type.
enum Color { Red, Green = 5, Blue };
enum class Small : unsigned char { A = 1, B = 200 };
enum Negative { Neg = -3 };

struct UsesEnum {
  int operator()() __qpu__ {
    Color c = Green;
    if (c == Green)
      return static_cast<int>(c) + Blue;
    return Red;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__UsesEnum() -> i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 5 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 6 : i32
// CHECK:           cc.store %[[VAL_5]], %{{.*}} : !cc.ptr<i32>
// CHECK:           arith.cmpi eq, %{{.*}}, %[[VAL_5]] : i32
// CHECK:           arith.addi %{{.*}}, %[[VAL_6]] : i32

struct ScopedEnum {
  int operator()() __qpu__ {
    Small s = Small::B;
    Negative n = Neg;
    if (s < Small::B)
      return 0;
    if (s == Small::B)
      return static_cast<int>(n);
    return static_cast<int>(Small::A);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ScopedEnum() -> i32
// CHECK-DAG:       arith.constant -56 : i8
// CHECK-DAG:       arith.constant -3 : i32
// CHECK:           cc.alloca i8
// CHECK:           arith.cmpi ult, %{{.*}}, %{{.*}} : i8
// CHECK:           arith.cmpi eq, %{{.*}}, %{{.*}} : i8

struct LocalEnum {
  int operator()() __qpu__ {
    enum Local { X = 7, Y };
    Local l = Y;
    return l;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__LocalEnum() -> i32
// CHECK:           arith.constant 8 : i32
// clang-format on
