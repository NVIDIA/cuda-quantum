// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --expand-quake-register-traversals --split-input-file %s | FileCheck %s

module {
  func.func @static_register_traversal() {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c3 = arith.constant 3 : i64
    %q = quake.alloca !quake.veq<3>
    %loop = cc.loop while ((%iv = %c0) -> (i64)) {
      %go = arith.cmpi slt, %iv, %c3 : i64
      cc.condition %go(%iv : i64)
    } do {
    ^bb0(%iv: i64):
      %ref = quake.extract_ref %q[%iv] : (!quake.veq<3>, i64) -> !quake.ref
      quake.h %ref : (!quake.ref) -> ()
      cc.continue %iv : i64
    } step {
    ^bb0(%iv: i64):
      %next = arith.addi %iv, %c1 : i64
      cc.continue %next : i64
    }
    return
  }
}

// CHECK-LABEL: func.func @static_register_traversal
// CHECK-NOT: cc.loop
// CHECK: quake.extract_ref {{.*}}[0]
// CHECK: quake.h
// CHECK: quake.extract_ref {{.*}}[1]
// CHECK: quake.h
// CHECK: quake.extract_ref {{.*}}[2]
// CHECK: quake.h
// CHECK-NOT: cf.br

// -----

module {
  func.func @dynamic_slice_traversal() {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %q = quake.alloca !quake.veq<2>
    %loop = cc.loop while ((%iv = %c0) -> (i64)) {
      %go = arith.cmpi slt, %iv, %c2 : i64
      cc.condition %go(%iv : i64)
    } do {
    ^bb0(%iv: i64):
      %slice = quake.subveq %q, %iv, %iv
          : (!quake.veq<2>, i64, i64) -> !quake.veq<1>
      quake.mz %slice : (!quake.veq<1>) -> !cc.sequence<!cc.measure_handle>
      cc.continue %iv : i64
    } step {
    ^bb0(%iv: i64):
      %next = arith.addi %iv, %c1 : i64
      cc.continue %next : i64
    }
    return
  }
}

// CHECK-LABEL: func.func @dynamic_slice_traversal
// CHECK-NOT: cc.loop
// CHECK: quake.subveq {{.*}}, 0, 0
// CHECK: quake.mz
// CHECK: quake.subveq {{.*}}, 1, 1
// CHECK: quake.mz
// CHECK-NOT: cf.br

// -----

module {
  func.func @parent_dependent_nested_traversal() {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c3 = arith.constant 3 : i64
    %c4 = arith.constant 4 : i64
    %q = quake.alloca !quake.veq<4>
    %outer = cc.loop while ((%oi = %c0) -> (i64)) {
      %outer_go = arith.cmpi slt, %oi, %c3 : i64
      cc.condition %outer_go(%oi : i64)
    } do {
    ^bb0(%oi: i64):
      %start = arith.addi %oi, %c1 : i64
      %inner = cc.loop while ((%ii = %start) -> (i64)) {
        %inner_go = arith.cmpi slt, %ii, %c4 : i64
        cc.condition %inner_go(%ii : i64)
      } do {
      ^bb0(%ii: i64):
        %ref = quake.extract_ref %q[%ii] : (!quake.veq<4>, i64) -> !quake.ref
        quake.x %ref : (!quake.ref) -> ()
        cc.continue %ii : i64
      } step {
      ^bb0(%ii: i64):
        %next = arith.addi %ii, %c1 : i64
        cc.continue %next : i64
      }
      cc.continue %oi : i64
    } step {
    ^bb0(%oi: i64):
      %next = arith.addi %oi, %c1 : i64
      cc.continue %next : i64
    }
    return
  }
}

// CHECK-LABEL: func.func @parent_dependent_nested_traversal
// CHECK-NOT: cc.loop
// CHECK-COUNT-6: quake.extract_ref
// CHECK-NOT: cf.br

// -----

module {
  func.func @keep_outer_repeat() {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %c1000000 = arith.constant 1000000 : i64
    %q = quake.alloca !quake.veq<2>
    %outer = cc.loop while ((%oi = %c0) -> (i64)) {
      %outer_go = arith.cmpi slt, %oi, %c1000000 : i64
      cc.condition %outer_go(%oi : i64)
    } do {
    ^bb0(%oi: i64):
      %inner = cc.loop while ((%ii = %c0) -> (i64)) {
        %inner_go = arith.cmpi slt, %ii, %c2 : i64
        cc.condition %inner_go(%ii : i64)
      } do {
      ^bb0(%ii: i64):
        %ref = quake.extract_ref %q[%ii] : (!quake.veq<2>, i64) -> !quake.ref
        quake.x %ref : (!quake.ref) -> ()
        cc.continue %ii : i64
      } step {
      ^bb0(%ii: i64):
        %next = arith.addi %ii, %c1 : i64
        cc.continue %next : i64
      }
      cc.continue %oi : i64
    } step {
    ^bb0(%oi: i64):
      %next = arith.addi %oi, %c1 : i64
      cc.continue %next : i64
    }
    return
  }
}

// CHECK-LABEL: func.func @keep_outer_repeat
// CHECK: cc.loop
// CHECK: quake.extract_ref {{.*}}[0]
// CHECK: quake.x
// CHECK: quake.extract_ref {{.*}}[1]
// CHECK: quake.x
// CHECK-NOT: cc.loop
// CHECK-NOT: cf.br
