// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | qlx-opt | FileCheck %s

// Round-trips the cross-patch fabric.cx / fabric.cz forms through the
// custom parser/printer. Covers both:
//   * the single-patch (intra) form, which must still print the
//     dialect-stripped `: <@code>` shorthand (byte-for-byte unchanged), and
//   * the two-patch (cross) form, which prints the functional type
//     `(!fabric.patch<@a>, !fabric.patch<@b>) -> (...)`.

fabric.code @sc { distance = 3 : i64, partitions = {data = 9 : i64, sx = 4 : i64, sz = 4 : i64} }

// CHECK-LABEL: fabric.gadget @intra
// CHECK-SAME: (%[[P:.*]]: !fabric.patch<@sc>)
fabric.gadget @intra(%p: !fabric.patch<@sc>) -> !fabric.patch<@sc> {
  // Single-patch form: schedule keyword + dialect-stripped result type.
  // CHECK: %[[V0:.*]] = fabric.cx %[[P]] sx -> data {schedule = "hx"} : <@sc>
  %0 = fabric.cx %p sx -> data {schedule = "hx"} : !fabric.patch<@sc>
  // CHECK: %[[V1:.*]] = fabric.cz %[[V0]] data -> sz {pairs = "0:0,1:1"} : <@sc>
  %1 = fabric.cz %0 data -> sz {pairs = "0:0,1:1"} : !fabric.patch<@sc>
  // CHECK: fabric.return %[[V1]] : !fabric.patch<@sc>
  fabric.return %1 : !fabric.patch<@sc>
}

// CHECK-LABEL: fabric.gadget @cross
// CHECK-SAME: (%[[A:.*]]: !fabric.patch<@sc>, %[[B:.*]]: !fabric.patch<@sc>)
fabric.gadget @cross(%a: !fabric.patch<@sc>, %b: !fabric.patch<@sc>)
    -> (!fabric.patch<@sc>, !fabric.patch<@sc>) {
  // Two-patch form: ctrl partition on patches[0], targ partition on
  // patches[1]; functional-type round-trip with explicit `pairs`.
  // CHECK: %[[CX:.*]]:2 = fabric.cx %[[A]], %[[B]] data -> data {pairs = "0:0,1:1"} : (!fabric.patch<@sc>, !fabric.patch<@sc>) -> (!fabric.patch<@sc>, !fabric.patch<@sc>)
  %a1, %b1 = fabric.cx %a, %b data -> data {pairs = "0:0,1:1"}
      : (!fabric.patch<@sc>, !fabric.patch<@sc>) -> (!fabric.patch<@sc>, !fabric.patch<@sc>)
  // CHECK: %[[CZ:.*]]:2 = fabric.cz %[[CX]]#0, %[[CX]]#1 data -> data {pairs = "0:0"} : (!fabric.patch<@sc>, !fabric.patch<@sc>) -> (!fabric.patch<@sc>, !fabric.patch<@sc>)
  %a2, %b2 = fabric.cz %a1, %b1 data -> data {pairs = "0:0"}
      : (!fabric.patch<@sc>, !fabric.patch<@sc>) -> (!fabric.patch<@sc>, !fabric.patch<@sc>)
  // CHECK: fabric.return %[[CZ]]#0, %[[CZ]]#1 : !fabric.patch<@sc>, !fabric.patch<@sc>
  fabric.return %a2, %b2 : !fabric.patch<@sc>, !fabric.patch<@sc>
}
