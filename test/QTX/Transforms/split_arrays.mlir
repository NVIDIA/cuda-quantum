// ========================================================================== //
// Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                 //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: cudaq-opt --qtx-split-arrays %s | FileCheck %s
// RUN: cudaq-opt --qtx-split-arrays %s | CircuitCheck %s

module {

  // CHECK-LABEL: qtx.circuit @test_dealloc_arrays
  // CHECK: %[[VAL_0:.*]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[VAL_1:.*]]:3 = array_split %[[VAL_0]] : !qtx.wire_array<3>
  // CHECK: %[[VAL_2:.*]] = alloca : !qtx.wire
  // CHECK: %[[VAL_3:.*]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[VAL_4:.*]]:3 = array_split %[[VAL_3]] : !qtx.wire_array<3>
  // CHECK: dealloc %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2, %[[VAL_2]], %[[VAL_4]]#0, %[[VAL_4]]#1, %[[VAL_4]]#2 : !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire
  qtx.circuit @test_dealloc_arrays() {
    %a0 = alloca : !qtx.wire_array<3>
    %w0 = alloca : !qtx.wire
    %a1 = alloca : !qtx.wire_array<3>
    dealloc %a0, %w0, %a1 : !qtx.wire_array<3>, !qtx.wire, !qtx.wire_array<3>
    return
  }

  // CHECK-LABEL: qtx.circuit @test_reset_arrays
  // CHECK: %[[VAL_0:.*]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[VAL_1:.*]]:3 = array_split %[[VAL_0]] : !qtx.wire_array<3>
  // CHECK: %[[VAL_2:.*]] = alloca : !qtx.wire
  // CHECK: %[[VAL_3:.*]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[VAL_4:.*]]:3 = array_split %[[VAL_3]] : !qtx.wire_array<3>
  // CHECK: %[[VAL_5:.*]]:7 = reset %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2, %[[VAL_2]], %[[VAL_4]]#0, %[[VAL_4]]#1, %[[VAL_4]]#2 : !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire
  // CHECK: dealloc %[[VAL_5]]#0, %[[VAL_5]]#1, %[[VAL_5]]#2, %[[VAL_5]]#3, %[[VAL_5]]#4, %[[VAL_5]]#5, %[[VAL_5]]#6 : !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire
  qtx.circuit @test_reset_arrays() {
    %a0 = alloca : !qtx.wire_array<3>
    %w0 = alloca : !qtx.wire
    %a1 = alloca : !qtx.wire_array<3>
    %a2, %w1, %a3 = reset %a0, %w0, %a1 : !qtx.wire_array<3>, !qtx.wire, !qtx.wire_array<3>
    dealloc %a2, %w1, %a3 : !qtx.wire_array<3>, !qtx.wire, !qtx.wire_array<3>
    return
  }

  // CHECK-LABEL: qtx.circuit @test_01
  // CHECK: %[[VAL_0:.*]] = arith.constant 0 : index
  // CHECK: %[[VAL_1:.*]] = arith.constant 1 : index
  // CHECK: %[[VAL_2:.*]] = arith.constant 2 : index
  // CHECK: %[[VAL_3:.*]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[VAL_4:.*]]:3 = array_split %[[VAL_3]] : !qtx.wire_array<3>
  // CHECK: dealloc %[[VAL_4]]#0, %[[VAL_4]]#1, %[[VAL_4]]#2 : !qtx.wire, !qtx.wire, !qtx.wire
  qtx.circuit @test_01() {
    // Borrowing and not using it
    %i = arith.constant 0 : index
    %j = arith.constant 1 : index
    %k = arith.constant 2 : index
    %a0 = alloca : !qtx.wire_array<3>
    %w0, %a1 = array_borrow %i from %a0 : index from !qtx.wire_array<3> -> !qtx.wire, !qtx.wire_array<3, dead = 1>
    %w1, %w2, %a2 = array_borrow %j, %k from %a1 : index, index from !qtx.wire_array<3, dead = 1> -> !qtx.wire, !qtx.wire, !qtx.wire_array<3, dead = 3>
    %a3 = array_yield %w1 to %a2 : !qtx.wire_array<3, dead = 3> -> !qtx.wire_array<3, dead = 2>
    %a4 = array_yield %w0, %w2 to %a3 : !qtx.wire_array<3, dead = 2> -> !qtx.wire_array<3>
    dealloc %a4 : !qtx.wire_array<3>
    return
  }

  // CHECK-LABEL: qtx.circuit @test_02
  // CHECK: %[[VAL_0:.*]] = arith.constant 0 : index
  // CHECK: %[[VAL_1:.*]] = arith.constant 1 : index
  // CHECK: %[[VAL_2:.*]] = arith.constant 2 : index
  // CHECK: %[[VAL_3:.*]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[VAL_4:.*]]:3 = array_split %[[VAL_3]] : !qtx.wire_array<3>
  // CHECK: %[[VAL_5:.*]] = h %[[VAL_4]]#0 : !qtx.wire
  // CHECK: %[[VAL_6:.*]] = h %[[VAL_4]]#1 : !qtx.wire
  // CHECK: %[[VAL_7:.*]] = h %[[VAL_4]]#2 : !qtx.wire
  // CHECK: dealloc %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : !qtx.wire, !qtx.wire, !qtx.wire
  qtx.circuit @test_02() {
    // Borrowing all wires
    %i = arith.constant 0 : index
    %j = arith.constant 1 : index
    %k = arith.constant 2 : index
    %a0 = alloca : !qtx.wire_array<3>
    %w0, %a1 = array_borrow %i from %a0 : index from !qtx.wire_array<3> -> !qtx.wire, !qtx.wire_array<3, dead = 1>
    %w1, %w2, %a2 = array_borrow %j, %k from %a1 : index, index from !qtx.wire_array<3, dead = 1> -> !qtx.wire, !qtx.wire, !qtx.wire_array<3, dead = 3>
    %w3 = h %w0 : !qtx.wire
    %w4 = h %w1 : !qtx.wire
    %w5 = h %w2 : !qtx.wire
    %a3 = array_yield %w3 to %a2 : !qtx.wire_array<3, dead = 3> -> !qtx.wire_array<3, dead = 2>
    %a4 = array_yield %w4, %w5 to %a3 : !qtx.wire_array<3, dead = 2> -> !qtx.wire_array<3>
    dealloc %a4 : !qtx.wire_array<3>
    return
  }

  // CHECK-LABEL: qtx.circuit @test_03
  // CHECK: %[[VAL_0:.*]] = arith.constant 0 : index
  // CHECK: %[[VAL_1:.*]] = arith.constant 1 : index
  // CHECK: %[[VAL_2:.*]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[VAL_3:.*]]:3 = array_split %[[VAL_2]] : !qtx.wire_array<3>
  // CHECK: %[[VAL_4:.*]] = h {{\[}}%[[VAL_3]]#1] %[[VAL_3]]#0 : [!qtx.wire] !qtx.wire
  // CHECK: %[[VAL_5:.*]]:2 = swap %[[VAL_4]], %[[VAL_3]]#1 : !qtx.wire, !qtx.wire
  // CHECK: %[[VAL_6:.*]] = h %[[VAL_5]]#1 : !qtx.wire
  // CHECK: dealloc %[[VAL_5]]#0, %[[VAL_6]], %[[VAL_3]]#2 : !qtx.wire, !qtx.wire, !qtx.wire
  qtx.circuit @test_03() {
    // Borrowing two wires using two borrow operations and one yield
    %i = arith.constant 0 : index
    %j = arith.constant 1 : index
    %a0 = alloca : !qtx.wire_array<3>
    %w0, %a1 = array_borrow %i from %a0 : index from !qtx.wire_array<3> -> !qtx.wire, !qtx.wire_array<3, dead = 1>
    %w1, %a2 = array_borrow %j from %a1 : index from !qtx.wire_array<3, dead = 1> -> !qtx.wire, !qtx.wire_array<3, dead = 2>
    %w2 = h [%w1] %w0 : [!qtx.wire] !qtx.wire
    %w3, %w4 = swap %w2, %w1 : !qtx.wire, !qtx.wire
    %w5 = h %w4 : !qtx.wire
    %a3 = array_yield %w3, %w5 to %a2 : !qtx.wire_array<3, dead = 2> -> !qtx.wire_array<3>
    dealloc %a3 : !qtx.wire_array<3>
    return
  }

  // CHECK-LABEL: qtx.circuit @test_04
  // CHECK: %[[VAL_0:.*]] = arith.constant 0 : index
  // CHECK: %[[VAL_1:.*]] = arith.constant 1 : index
  // CHECK: %[[VAL_2:.*]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[VAL_3:.*]]:3 = array_split %[[VAL_2]] : !qtx.wire_array<3>
  // CHECK: %[[VAL_4:.*]] = h %[[VAL_3]]#0 : !qtx.wire
  // CHECK: %[[VAL_5:.*]]:2 = swap %[[VAL_4]], %[[VAL_3]]#1 : !qtx.wire, !qtx.wire
  // CHECK: %[[VAL_6:.*]] = h %[[VAL_5]]#1 : !qtx.wire
  // CHECK: dealloc %[[VAL_5]]#0, %[[VAL_6]], %[[VAL_3]]#2 : !qtx.wire, !qtx.wire, !qtx.wire
  qtx.circuit @test_04() {
    // Borrowing two wires using one borrow operation and two yields
    %i = arith.constant 0 : index
    %j = arith.constant 1 : index
    %a0 = alloca : !qtx.wire_array<3>
    %w0, %w1, %a1 = array_borrow %i, %j from %a0 : index, index from !qtx.wire_array<3> -> !qtx.wire, !qtx.wire, !qtx.wire_array<3, dead = 2>

    %w2 = h %w0 : !qtx.wire
    %w3, %w4 = swap %w2, %w1 : !qtx.wire, !qtx.wire
    %w5 = h %w4 : !qtx.wire

    %a2 = array_yield %w3 to %a1 : !qtx.wire_array<3, dead = 2> -> !qtx.wire_array<3, dead = 1>
    %a3 = array_yield %w5 to %a2 : !qtx.wire_array<3, dead = 1> -> !qtx.wire_array<3>
    dealloc %a3 : !qtx.wire_array<3>
    return
  }

  // CHECK-LABEL: qtx.circuit @test_already_split_base_array
  // CHECK: %[[VAL_0:.*]] = alloca : !qtx.wire_array<2>
  // CHECK: %[[VAL_1:.*]] = alloca : !qtx.wire
  // CHECK: %[[VAL_2:.*]] = h %[[VAL_1]] : !qtx.wire
  // CHECK: %[[VAL_3:.*]]:2 = array_split %[[VAL_0]] : !qtx.wire_array<2>
  // CHECK: %[[VAL_4:.*]] = h %[[VAL_3]]#0 : !qtx.wire
  // CHECK: %[[VAL_5:.*]] = h %[[VAL_3]]#1 : !qtx.wire
  // CHECK: dealloc %[[VAL_2]], %[[VAL_4]], %[[VAL_5]] : !qtx.wire, !qtx.wire, !qtx.wire
  qtx.circuit @test_already_split_base_array() {
    // This test in making sure that the pass don't split `%a0`---the code already does that.
    %a0 = alloca : !qtx.wire_array<2>
    %w0 = alloca : !qtx.wire
    %w1 = h %w0 : !qtx.wire
    %w2, %w3 = array_split %a0 : !qtx.wire_array<2>
    %w4 = h %w2 : !qtx.wire
    %w5 = h %w3 : !qtx.wire
    dealloc %w1, %w4, %w5 : !qtx.wire, !qtx.wire, !qtx.wire
    return
  }

  // CHECK-LABEL: qtx.circuit @test_split_derived_array
  // CHECK: %[[VAL_0:.*]] = arith.constant 0 : index
  // CHECK: %[[VAL_1:.*]] = alloca : !qtx.wire_array<2>
  // CHECK: %[[VAL_2:.*]]:2 = array_split %[[VAL_1]] : !qtx.wire_array<2>
  // CHECK: %[[VAL_3:.*]] = h %[[VAL_2]]#0 : !qtx.wire
  // CHECK: %[[VAL_4:.*]] = h %[[VAL_3]] : !qtx.wire
  // CHECK: %[[VAL_5:.*]] = h %[[VAL_2]]#1 : !qtx.wire
  // CHECK: dealloc %[[VAL_4]], %[[VAL_5]] : !qtx.wire, !qtx.wire
  qtx.circuit @test_split_derived_array() {
    %i = arith.constant 0 : index
    %a0 = alloca : !qtx.wire_array<2>
    %w0, %a1 = array_borrow %i from %a0 : index from !qtx.wire_array<2> -> !qtx.wire, !qtx.wire_array<2, dead = 1>
    %w1 = h %w0 : !qtx.wire
    %a2 = array_yield %w1 to %a1 : !qtx.wire_array<2, dead = 1> -> !qtx.wire_array<2>
    %w2, %w3 = array_split %a2 : !qtx.wire_array<2>
    %w4 = h %w2 : !qtx.wire
    %w5 = h %w3 : !qtx.wire
    dealloc %w4, %w5 : !qtx.wire, !qtx.wire
    return
  }

  // CHECK-LABEL: qtx.circuit @test_measure
  // CHECK: %[[VAL_0:.*]] = alloca : !qtx.wire_array<2>
  // CHECK: %[[VAL_1:.*]]:2 = array_split %[[VAL_0]] : !qtx.wire_array<2>
  // CHECK: %[[VAL_2:.*]] = alloca : !qtx.wire
  // CHECK: %[[VAL_3:.*]] = alloca : !qtx.wire_array<3>
  // CHECK: %[[VAL_4:.*]]:3 = array_split %[[VAL_3]] : !qtx.wire_array<3>
  // CHECK: %[[VAL_5:.*]], %[[VAL_6:.*]]:6 = mx %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_2]], %[[VAL_4]]#0, %[[VAL_4]]#1, %[[VAL_4]]#2 : !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire -> <vector<6xi1>> !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire
  // CHECK: dealloc %[[VAL_6]]#0, %[[VAL_6]]#1, %[[VAL_6]]#2, %[[VAL_6]]#3, %[[VAL_6]]#4, %[[VAL_6]]#5 : !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire
  qtx.circuit @test_measure() {
    %a0 = alloca : !qtx.wire_array<2>
    %w0 = alloca : !qtx.wire
    %a1 = alloca : !qtx.wire_array<3>
    %reg, %a2, %w1, %a3 = mx %a0, %w0, %a1 : !qtx.wire_array<2>, !qtx.wire, !qtx.wire_array<3> -> <vector<6xi1>> !qtx.wire_array<2>, !qtx.wire, !qtx.wire_array<3>
    dealloc %a2, %w1, %a3 : !qtx.wire_array<2>, !qtx.wire, !qtx.wire_array<3>
    return
  }
}
