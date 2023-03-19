// ========================================================================== //
// Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                 //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: cudaq-opt %s | cudaq-opt | FileCheck %s

module {

  // CHECK-LABEL: @array_create
  // CHECK: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK: %[[W1:.+]] = alloca : !qtx.wire
  // CHECK: %[[W2:.+]] = alloca : !qtx.wire
  // CHECK: %[[ARRAY:.+]] = array_create %[[W0]], %[[W1]], %[[W2]] : !qtx.wire_array<3>
  // CHECK: dealloc %[[ARRAY]] : !qtx.wire_array<3>
  qtx.circuit @array_create() {
    %w0 = alloca : !qtx.wire
    %w1 = alloca : !qtx.wire
    %w2 = alloca : !qtx.wire
    %array = array_create %w0, %w1, %w2 : !qtx.wire_array<3>
    dealloc %array : !qtx.wire_array<3>
    return
  }

  // CHECK-LABEL: @array_split
  // CHECK: %[[A:.+]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[W:.+]]:3 = array_split %[[A]] : !qtx.wire_array<3>
  // CHECK: dealloc %[[W]]#0 : !qtx.wire
  // CHECK: dealloc %[[W]]#1 : !qtx.wire
  // CHECK: dealloc %[[W]]#2 : !qtx.wire
  qtx.circuit @array_split() {
    %array = alloca : !qtx.wire_array<3>
    %w0, %w1, %w2 = array_split %array : !qtx.wire_array<3>
    dealloc %w0 : !qtx.wire
    dealloc %w1 : !qtx.wire
    dealloc %w2 : !qtx.wire
    return
  }

  // CHECK-LABEL: @array_borrow_yield
  // CHECK: %[[I:.+]] = arith.constant
  // CHECK: %[[J:.+]] = arith.constant
  // CHECK: %[[K:.+]] = arith.constant
  // CHECK: %[[A0:.+]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[W0:.+]], %[[A1:.+]] = array_borrow %[[I]] from %[[A0]] : index from !qtx.wire_array<3> -> !qtx.wire, !qtx.wire_array<3, dead = 1>
  // CHECK: %[[W12:.+]]:2, %[[A2:.+]] = array_borrow %[[J]], %[[K]] from %[[A1]] : i32, i64 from !qtx.wire_array<3, dead = 1> -> !qtx.wire, !qtx.wire, !qtx.wire_array<3, dead = 3>
  // CHECK: %[[A3:.+]] = array_yield %[[W12]]#0 to %[[A2]] : !qtx.wire_array<3, dead = 3> -> !qtx.wire_array<3, dead = 2>
  // CHECK: %[[A4:.+]] = array_yield %[[W0]], %[[W12]]#1 to %[[A3]] : !qtx.wire_array<3, dead = 2> -> !qtx.wire_array<3>
  // CHECK: dealloc %[[A4]] : !qtx.wire_array<3>
  qtx.circuit @array_borrow_yield() {
    %i = arith.constant 0 : index
    %j = arith.constant 1 : i32
    %k = arith.constant 2 : i64
    %a0 = alloca : !qtx.wire_array<3>
    %w0, %a1 = array_borrow %i from %a0 : index from !qtx.wire_array<3> -> !qtx.wire, !qtx.wire_array<3, dead = 1>
    %w1, %w2, %a2 = array_borrow %j, %k from %a1 : i32, i64 from !qtx.wire_array<3, dead = 1> -> !qtx.wire, !qtx.wire, !qtx.wire_array<3, dead = 3>
    %a3 = array_yield %w1 to %a2 : !qtx.wire_array<3, dead = 3> -> !qtx.wire_array<3, dead = 2>
    %a4 = array_yield %w0, %w2 to %a3 : !qtx.wire_array<3, dead = 2> -> !qtx.wire_array<3>
    dealloc %a4 : !qtx.wire_array<3>
    return
  }

  // CHECK-LABEL: @mz_array
  // CHECK: %[[A0:.+]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[BITS:.+]], %[[A1:.+]] = mz %[[A0]] : !qtx.wire_array<3> -> <vector<3xi1>> !qtx.wire_array<3>
  // CHECK: dealloc %[[A1]] : !qtx.wire_array<3>
  qtx.circuit @mz_array() {
    %array = alloca : !qtx.wire_array<3>
    %bits, %array_1 = mz %array : !qtx.wire_array<3> -> <vector<3xi1>> !qtx.wire_array<3>
    dealloc %array_1 : !qtx.wire_array<3>
    return
  }

  // CHECK-LABEL: @mz_array_size_one
  // CHECK: %[[A0:.+]] = alloca : !qtx.wire_array<1>
  // CHECK: %[[BITS:.+]], %[[A1:.+]] = mz %[[A0]] : !qtx.wire_array<1> -> <i1> !qtx.wire_array<1>
  // CHECK: dealloc %[[A1]] : !qtx.wire_array<1>
  qtx.circuit @mz_array_size_one() {
    %array = alloca : !qtx.wire_array<1>
    %bits, %array_1 = mz %array : !qtx.wire_array<1> -> <i1> !qtx.wire_array<1>
    dealloc %array_1 : !qtx.wire_array<1>
    return
  }

  // CHECK-LABEL: @mz_array_size_one_vector
  // CHECK: %[[A0:.+]] = alloca : !qtx.wire_array<1>
  // CHECK: %[[BITS:.+]], %[[A1:.+]] = mz %[[A0]] : !qtx.wire_array<1> -> <vector<1xi1>> !qtx.wire_array<1>
  // CHECK: dealloc %[[A1]] : !qtx.wire_array<1>
  qtx.circuit @mz_array_size_one_vector() {
    %array = alloca : !qtx.wire_array<1>
    %bits, %array_1 = mz %array : !qtx.wire_array<1> -> <vector<1xi1>> !qtx.wire_array<1>
    dealloc %array_1 : !qtx.wire_array<1>
    return
  }

  // CHECK-LABEL: @mz_array_and_wire
  // CHECK: %[[A0:.+]] = alloca : !qtx.wire_array<2>
  // CHECK: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK: %[[BITS:.+]], %[[A_W:.+]]:2 = mz %[[A0]], %[[W0]] : !qtx.wire_array<2>, !qtx.wire -> <vector<3xi1>> !qtx.wire_array<2>, !qtx.wire
  // CHECK: dealloc %[[A_W]]#0, %[[A_W]]#1 : !qtx.wire_array<2>, !qtx.wire
  qtx.circuit @mz_array_and_wire() {
    %array = alloca : !qtx.wire_array<2>
    %w0 = alloca : !qtx.wire
    %bits, %array_1, %w1 = mz %array, %w0: !qtx.wire_array<2>, !qtx.wire -> <vector<3xi1>> !qtx.wire_array<2>, !qtx.wire
    dealloc %array_1, %w1: !qtx.wire_array<2>, !qtx.wire
    return
  }

  // CHECK-LABEL: @mz_split_array
  // CHECK: %[[A0:.+]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[W0:.+]]:3 = array_split %[[A0]] : !qtx.wire_array<3>
  // CHECK: %[[REG:.+]], %[[W1:.+]]:3 = mz %[[W0]]#0, %[[W0]]#1, %[[W0]]#2 : !qtx.wire, !qtx.wire, !qtx.wire -> <vector<3xi1>> !qtx.wire, !qtx.wire, !qtx.wire
  // CHECK: dealloc %[[W1]]#0, %[[W1]]#1, %[[W1]]#2 : !qtx.wire, !qtx.wire, !qtx.wire
  qtx.circuit @mz_split_array() {
    %array = alloca : !qtx.wire_array<3>
    %w0, %w1, %w2 = array_split %array : !qtx.wire_array<3>
    %reg, %w3, %w4, %w5 = mz %w0, %w1, %w2 : !qtx.wire, !qtx.wire, !qtx.wire -> <vector<3xi1>> !qtx.wire, !qtx.wire, !qtx.wire
    dealloc %w3, %w4, %w5 : !qtx.wire, !qtx.wire, !qtx.wire
    return
  }

  // CHECK-LABEL: @reset_array
  // CHECK: %[[A0:.+]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[A1:.+]] = reset %[[A0]] : !qtx.wire_array<3>
  // CHECK: dealloc %[[A1]] : !qtx.wire_array<3>
  qtx.circuit @reset_array() {
    %array = alloca : !qtx.wire_array<3>
    %array_1 = reset %array : !qtx.wire_array<3>
    dealloc %array_1 : !qtx.wire_array<3>
    return
  }
}
