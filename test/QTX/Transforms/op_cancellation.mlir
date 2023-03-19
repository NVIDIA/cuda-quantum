// ========================================================================== //
// Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                 //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: cudaq-opt --qtx-op-cancellation %s | FileCheck %s

module {
  // CHECK-LABEL: @trivial_one_qubit_ops
  // CHECK-NEXT: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: dealloc %[[W0]] : !qtx.wire
  qtx.circuit @trivial_one_qubit_ops() {
    %w0 = alloca : !qtx.wire
    %w1 = x %w0 : !qtx.wire
    %w2 = x %w1 : !qtx.wire

    // Corner case that we might not run into.  Since `h` is Hermitian the
    // `adj` property should not make a differnce.
    %w3 = h %w2 : !qtx.wire
    %w4 = h<adj> %w3 : !qtx.wire

    %w5 = t %w4 : !qtx.wire
    %w6 = t<adj> %w5 : !qtx.wire
    dealloc %w6 : !qtx.wire
    return
  }

  // CHECK-LABEL: @trivial_one_qubit_ops_not_messing_up_adj
  // CHECK-NEXT: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[W1:.+]] = t<adj> %[[W0]] : !qtx.wire
  // CHECK-NEXT: %[[W2:.+]] = t<adj> %[[W1]] : !qtx.wire
  // CHECK-NEXT: dealloc %[[W2]] : !qtx.wire
  qtx.circuit @trivial_one_qubit_ops_not_messing_up_adj() {
    %w0 = alloca : !qtx.wire
    %w1 = t<adj> %w0 : !qtx.wire
    %w2 = t<adj> %w1 : !qtx.wire
    dealloc %w2 : !qtx.wire
    return
  }

  // CHECK-LABEL: @trivial_one_qubit_params_ops
  // CHECK-NEXT: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: dealloc %[[W0]] : !qtx.wire
  qtx.circuit @trivial_one_qubit_params_ops() {
    %w0 = alloca : !qtx.wire
    %angle = arith.constant 3.14 : f64
    %w1 = r1<%angle> %w0 : <f64> !qtx.wire
    %w2 = r1<adj, %angle> %w1 : <f64> !qtx.wire
    dealloc %w2 : !qtx.wire
    return
  }

  // CHECK-LABEL: @corner_case_0_one_qubit_params_ops
  // CHECK-NEXT: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: dealloc %[[W0]] : !qtx.wire
  qtx.circuit @corner_case_0_one_qubit_params_ops() {
    %w0 = alloca : !qtx.wire
    // Note: constant folding will kick-in
    %angle0 = arith.constant 3.14 : f64
    %angle1 = arith.constant 3.14 : f64
    %w1 = r1<%angle0> %w0 : <f64> !qtx.wire
    %w2 = r1<adj, %angle1> %w1 : <f64> !qtx.wire
    dealloc %w2 : !qtx.wire
    return
  }

  // CHECK-LABEL: @corner_case_1_one_qubit_params_ops
  // CHECK-DAG: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK-DAG: %[[A0:.+]] = arith.constant 3.14{{.*}} : f64
  // CHECK-DAG: %[[A1:.+]] = arith.constant 3.14{{.*}} : f32
  // CHECK-NEXT: %[[W1:.+]] = r1<%[[A0]]> %[[W0]] : <f64> !qtx.wire
  // CHECK-NEXT: %[[W2:.+]] = r1<adj, %[[A1]]> %[[W1]] : <f32> !qtx.wire
  // CHECK-NEXT: dealloc %[[W2]] : !qtx.wire
  qtx.circuit @corner_case_1_one_qubit_params_ops() {
    %w0 = alloca : !qtx.wire
    %a0 = arith.constant 3.14 : f64
    %a1 = arith.constant 3.14 : f32
    %w1 = r1<%a0> %w0 : <f64> !qtx.wire
    %w2 = r1<adj, %a1> %w1 : <f32> !qtx.wire
    dealloc %w2 : !qtx.wire
    return
  }

  // CHECK-LABEL: @corner_case_2_one_qubit_params_ops
  // CHECK-DAG: %[[WIRE_0:.+]] = alloca : !qtx.wire
  // CHECK-DAG: %[[ANGLE0:.+]] = arith.constant 3.14
  // CHECK-DAG: %[[ANGLE1:.+]] = arith.constant 6.28
  // CHECK-NEXT: %[[WIRE_1:.+]] = r1<%[[ANGLE0]]> %[[WIRE_0]] : <f64> !qtx.wire
  // CHECK-NEXT: %[[WIRE_2:.+]] = r1<adj, %[[ANGLE1]]> %[[WIRE_1]] : <f64> !qtx.wire
  // CHECK-NEXT: dealloc %[[WIRE_2]] : !qtx.wire
  qtx.circuit @corner_case_2_one_qubit_params_ops() {
    %w_0 = alloca : !qtx.wire
    %angle0 = arith.constant 3.14 : f64
    %angle1 = arith.constant 6.28 : f64
    %w_1 = r1<%angle0> %w_0 : <f64> !qtx.wire
    %w_2 = r1<adj, %angle1> %w_1 : <f64> !qtx.wire
    dealloc %w_2 : !qtx.wire
    return
  }

  // CHECK-LABEL: @trivial_one_qubit_controlled_ops
  // CHECK-NEXT: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[C:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: dealloc %[[W0]] : !qtx.wire
  // CHECK-NEXT: dealloc %[[C]] : !qtx.wire
  qtx.circuit @trivial_one_qubit_controlled_ops() {
    %w0 = alloca : !qtx.wire
    %c = alloca : !qtx.wire
    %w1 = x [%c] %w0 : [!qtx.wire] !qtx.wire
    %w2 = x [%c] %w1 : [!qtx.wire] !qtx.wire
    %w3 = t [%c] %w2 : [!qtx.wire] !qtx.wire
    %w4 = t<adj> [%c] %w3 : [!qtx.wire] !qtx.wire
    dealloc %w4 : !qtx.wire
    dealloc %c : !qtx.wire
    return
  }

  // CHECK-LABEL: @trivial_one_qubit_params_controlled_ops
  // CHECK-NEXT: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[C:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: dealloc %[[W0]] : !qtx.wire
  // CHECK-NEXT: dealloc %[[C]] : !qtx.wire
  qtx.circuit @trivial_one_qubit_params_controlled_ops() {
    %a = arith.constant 3.14 : f64
    %w0 = alloca : !qtx.wire
    %c = alloca : !qtx.wire
    %w1 = r1<%a> [%c] %w0 : <f64> [!qtx.wire] !qtx.wire
    %w2 = r1<adj, %a> [%c] %w1 : <f64> [!qtx.wire] !qtx.wire
    dealloc %w2 : !qtx.wire
    dealloc %c : !qtx.wire
    return
  }

  // CHECK-LABEL: @commutative_one_qubit_controlled_ops
  // CHECK-NEXT: %[[W0_0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[W1_0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[C:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[W1_1:.+]] = x [%[[C]]] %[[W1_0]] : [!qtx.wire] !qtx.wire
  // CHECK-NEXT: dealloc %[[W0_0]] : !qtx.wire
  // CHECK-NEXT: dealloc %[[W1_1]] : !qtx.wire
  // CHECK-NEXT: dealloc %[[C]] : !qtx.wire
  qtx.circuit @commutative_one_qubit_controlled_ops() {
    %w0 = alloca : !qtx.wire
    %w1 = alloca : !qtx.wire
    %c = alloca : !qtx.wire

    %w2 = x [%c] %w0 : [!qtx.wire] !qtx.wire
    %w3 = x [%c] %w1 : [!qtx.wire] !qtx.wire
    %w4 = x [%c] %w2 : [!qtx.wire] !qtx.wire

    dealloc %w4 : !qtx.wire
    dealloc %w3 : !qtx.wire
    dealloc %c : !qtx.wire
    return
  }

  // CHECK-LABEL: @fancy_commutative_one_qubit_controlled_ops
  // CHECK-NEXT: %[[W0_0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[C0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[C1:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[W0_1:.+]] = x [%[[C0]]] %[[W0_0]] : [!qtx.wire] !qtx.wire
  // CHECK-NEXT: %[[W0_2:.+]] = x [%[[C1]]] %[[W0_1]] : [!qtx.wire] !qtx.wire
  // CHECK-NEXT: %[[W0_3:.+]] = x [%[[C0]]] %[[W0_2]] : [!qtx.wire] !qtx.wire
  // CHECK-NEXT: dealloc %[[W0_3]] : !qtx.wire
  // CHECK-NEXT: dealloc %[[C0]] : !qtx.wire
  // CHECK-NEXT: dealloc %[[C1]] : !qtx.wire
  qtx.circuit @fancy_commutative_one_qubit_controlled_ops() {
    %w0_0 = alloca : !qtx.wire
    %c0 = alloca : !qtx.wire
    %c1 = alloca : !qtx.wire

    // The first and last `x` operations would cancel out if we were using
    // "fancier" commutation rules
    %w0_1 = x [%c0] %w0_0 : [!qtx.wire] !qtx.wire
    %w0_2 = x [%c1] %w0_1 : [!qtx.wire] !qtx.wire
    %w0_3 = x [%c0] %w0_2 : [!qtx.wire] !qtx.wire

    dealloc %w0_3 : !qtx.wire
    dealloc %c0 : !qtx.wire
    dealloc %c1 : !qtx.wire
    return
  }

  // CHECK-LABEL: @trivial_two_qubit_ops
  // CHECK-NEXT: %[[W0_0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[W1_0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: dealloc %[[W0_0]] : !qtx.wire
  // CHECK-NEXT: dealloc %[[W1_0]] : !qtx.wire
  qtx.circuit @trivial_two_qubit_ops() {
    %w0_0 = alloca : !qtx.wire
    %w1_0 = alloca : !qtx.wire

    %w0_1, %w1_1 = swap %w0_0, %w1_0 : !qtx.wire, !qtx.wire
    %w0_2, %w1_2 = swap %w0_1, %w1_1 : !qtx.wire, !qtx.wire

    dealloc %w0_2 : !qtx.wire
    dealloc %w1_2 : !qtx.wire
    return
  }


  // CHECK-LABEL: @corner_case_1_two_qubit_ops
  // CHECK-NEXT: %[[W0_0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[W1_0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[W2_0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[VAR0:[0-9]+]]:2 = swap %[[W0_0]], %[[W1_0]] : !qtx.wire, !qtx.wire
  // CHECK-NEXT: %[[VAR1:[0-9]+]]:2 = swap %[[VAR0]]#0, %[[W2_0]] : !qtx.wire, !qtx.wire
  // CHECK-NEXT: dealloc %[[VAR1]]#0 : !qtx.wire
  // CHECK-NEXT: dealloc %[[VAR0]]#1 : !qtx.wire
  // CHECK-NEXT: dealloc %[[VAR1]]#1 : !qtx.wire
  qtx.circuit @corner_case_1_two_qubit_ops() {
    %w0_0 = alloca : !qtx.wire
    %w1_0 = alloca : !qtx.wire
    %w2_0 = alloca : !qtx.wire

    %w0_1, %w1_1 = swap %w0_0, %w1_0 : !qtx.wire, !qtx.wire
    %w0_2, %w2_1 = swap %w0_1, %w2_0 : !qtx.wire, !qtx.wire

    dealloc %w0_2 : !qtx.wire
    dealloc %w1_1 : !qtx.wire
    dealloc %w2_1 : !qtx.wire
    return
  }

  // CHECK-LABEL: @corner_case_2_two_qubit_ops
  // CHECK-NEXT: %[[W0_0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: %[[W1_0:.+]] = alloca : !qtx.wire
  // CHECK-NEXT: dealloc %[[W0_0]] : !qtx.wire
  // CHECK-NEXT: dealloc %[[W1_0]] : !qtx.wire
  qtx.circuit @corner_case_2_two_qubit_ops() {
    %w0_0 = alloca : !qtx.wire
    %w1_0 = alloca : !qtx.wire

    %w0_1 = x %w0_0 : !qtx.wire
    %w0_2, %w1_1 = swap %w0_1, %w1_0 : !qtx.wire, !qtx.wire
    %w0_3, %w1_2 = swap %w0_2, %w1_1 : !qtx.wire, !qtx.wire
    %w0_4 = x %w0_3 : !qtx.wire

    dealloc %w0_4 : !qtx.wire
    dealloc %w1_2 : !qtx.wire
    return
  }
}
