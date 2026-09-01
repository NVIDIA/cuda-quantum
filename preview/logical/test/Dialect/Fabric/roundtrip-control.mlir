// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | qlx-opt | FileCheck %s

fabric.code @sc { distance = 3 : i64, partitions = {data = 9 : i64, sx = 4 : i64, sz = 4 : i64} }

// CHECK-LABEL: fabric.gadget @barrier_test
fabric.gadget @barrier_test(
    %a: !fabric.patch<@sc>, %b: !fabric.patch<@sc>)
    -> (!fabric.patch<@sc>, !fabric.patch<@sc>) {
  // CHECK: fabric.barrier
  %a1, %b1 = fabric.barrier %a, %b : !fabric.patch<@sc>, !fabric.patch<@sc>
      -> !fabric.patch<@sc>, !fabric.patch<@sc>
  fabric.return %a1, %b1 : !fabric.patch<@sc>, !fabric.patch<@sc>
}

// CHECK-LABEL: fabric.gadget @idle_test
fabric.gadget @idle_test(%p: !fabric.patch<@sc>) -> !fabric.patch<@sc> {
  // CHECK: fabric.idle %{{.*}} {rounds = 3 : i64}
  %0 = fabric.idle %p {rounds = 3 : i64} : !fabric.patch<@sc>
  fabric.return %0 : !fabric.patch<@sc>
}

// CHECK-LABEL: func.func @if_test
func.func @if_test(%cond: i1, %p0: !fabric.patch<@sc>, %p1: !fabric.patch<@sc>)
    -> !fabric.patch<@sc> {
  // CHECK: fabric.if
  %r = fabric.if %cond -> !fabric.patch<@sc> {
    fabric.yield %p0 : !fabric.patch<@sc>
  } else {
    fabric.yield %p1 : !fabric.patch<@sc>
  }
  return %r : !fabric.patch<@sc>
}

// CHECK-LABEL: func.func @xor_test
func.func @xor_test(%a: i1, %b: i1) -> i1 {
  // CHECK: fabric.xor
  %r = fabric.xor %a, %b : i1
  return %r : i1
}
