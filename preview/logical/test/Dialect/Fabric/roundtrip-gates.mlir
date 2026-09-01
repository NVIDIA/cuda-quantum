// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | qlx-opt | FileCheck %s

fabric.code @sc {
  distance = 3 : i64,
  partitions = {data = 9 : i64, sx = 4 : i64, sz = 4 : i64},
  hx = [array<i64: 0>]
}

// CHECK-LABEL: fabric.gadget @gate_ops
// CHECK-SAME: (%[[P:.*]]: !fabric.patch<@sc>)
fabric.gadget @gate_ops(%p: !fabric.patch<@sc>) -> !fabric.patch<@sc> {
  // CHECK: %[[V0:.*]] = fabric.h %[[P]] data  : !fabric.patch<@sc>
  %0 = fabric.h %p data : !fabric.patch<@sc>
  // CHECK: %[[V1:.*]] = fabric.s %[[V0]] sx  : !fabric.patch<@sc>
  %1 = fabric.s %0 sx : !fabric.patch<@sc>
  // CHECK: %[[V2:.*]] = fabric.sdg %[[V1]] sz  : !fabric.patch<@sc>
  %2 = fabric.sdg %1 sz : !fabric.patch<@sc>
  // CHECK: %[[V3:.*]] = fabric.x %[[V2]] all  : !fabric.patch<@sc>
  %3 = fabric.x %2 all : !fabric.patch<@sc>
  // CHECK: %[[V4:.*]] = fabric.z %[[V3]] data  : !fabric.patch<@sc>
  %4 = fabric.z %3 data : !fabric.patch<@sc>
  // CHECK: %[[V5:.*]] = fabric.t %[[V4]] data  : !fabric.patch<@sc>
  %5 = fabric.t %4 data : !fabric.patch<@sc>
  // CHECK: %[[V6:.*]] = fabric.tdg %[[V5]] data  : !fabric.patch<@sc>
  %6 = fabric.tdg %5 data : !fabric.patch<@sc>
  // CHECK: %[[V7:.*]] = fabric.reset %[[V6]] all  : !fabric.patch<@sc>
  %7 = fabric.reset %6 all : !fabric.patch<@sc>
  // CHECK: fabric.return %[[V7]] : !fabric.patch<@sc>
  fabric.return %7 : !fabric.patch<@sc>
}

// CHECK-LABEL: fabric.gadget @gate_with_indices
// CHECK-SAME: (%[[P:.*]]: !fabric.patch<@sc>)
fabric.gadget @gate_with_indices(%p: !fabric.patch<@sc>) -> !fabric.patch<@sc> {
  // CHECK: %[[V0:.*]] = fabric.h %[[P]] data [0, 2, 4] : !fabric.patch<@sc>
  %0 = fabric.h %p data [0, 2, 4] : !fabric.patch<@sc>
  // CHECK: fabric.return %[[V0]] : !fabric.patch<@sc>
  fabric.return %0 : !fabric.patch<@sc>
}

// CHECK-LABEL: fabric.gadget @two_qubit_ops
// CHECK-SAME: (%[[P:.*]]: !fabric.patch<@sc>)
fabric.gadget @two_qubit_ops(%p: !fabric.patch<@sc>) -> !fabric.patch<@sc> {
  // CHECK: %[[V0:.*]] = fabric.cx %[[P]] sx -> data {schedule = "hx"} : <@sc>
  %0 = fabric.cx %p sx -> data {schedule = "hx"} : !fabric.patch<@sc>
  // CHECK: %[[V1:.*]] = fabric.cz %[[V0]] data -> sz {pairs = "0:0,1:1"} : <@sc>
  %1 = fabric.cz %0 data -> sz {pairs = "0:0,1:1"} : !fabric.patch<@sc>
  // CHECK: fabric.return %[[V1]] : !fabric.patch<@sc>
  fabric.return %1 : !fabric.patch<@sc>
}

// CHECK-LABEL: fabric.gadget @transversal
// CHECK-SAME: (%[[A:.*]]: !fabric.patch<@sc>, %[[B:.*]]: !fabric.patch<@sc>)
fabric.gadget @transversal(%a: !fabric.patch<@sc>, %b: !fabric.patch<@sc>)
    -> (!fabric.patch<@sc>, !fabric.patch<@sc>) {
  // CHECK: %[[V0:.*]], %[[V1:.*]] = fabric.transversal_cx %[[A]], %[[B]]
  %0, %1 = fabric.transversal_cx %a, %b : (!fabric.patch<@sc>, !fabric.patch<@sc>) -> (!fabric.patch<@sc>, !fabric.patch<@sc>)
  // CHECK: fabric.return %[[V0]], %[[V1]] : !fabric.patch<@sc>, !fabric.patch<@sc>
  fabric.return %0, %1 : !fabric.patch<@sc>, !fabric.patch<@sc>
}
