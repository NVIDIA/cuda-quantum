// ========================================================================== //
// Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                 //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: cudaq-opt %s | cudaq-opt | FileCheck %s

module {

  // CHECK-LABEL: qtx.circuit @return_wires
  // CHECK-SAME: (%[[C0:.+]]: !qtx.wire, %[[C1:.+]]: !qtx.wire, %[[T:.+]]: !qtx.wire)
  // CHECK: %[[T1:.+]] = x [%[[C0]], %[[C1]]] %[[T]] : [!qtx.wire, !qtx.wire] !qtx.wire
  // CHECK: return %[[C0]], %[[C1]], %[[T1]] : !qtx.wire, !qtx.wire, !qtx.wire
  qtx.circuit @return_wires(%c0: !qtx.wire, %c1: !qtx.wire, %t: !qtx.wire) -> (!qtx.wire, !qtx.wire, !qtx.wire) {
    %t1 = x [%c0, %c1] %t : [!qtx.wire, !qtx.wire] !qtx.wire
    return %c0, %c1, %t1 : !qtx.wire, !qtx.wire, !qtx.wire
  }

  // CHECK-LABEL: qtx.circuit @return_classical
  // CHECK-SAME: () -> <i32>
  // CHECK: %[[C:.+]] = arith.constant
  // CHECK: return <%[[C]]> : <i32>
  qtx.circuit @return_classical() -> <i32> {
    %c = arith.constant 42 : i32
    return <%c> : <i32>
  }

  // CHECK-LABEL: qtx.circuit @return_both
  // CHECK-SAME: (%[[W0:.+]]: !qtx.wire)
  // CHECK: %[[C:.+]] = arith.constant
  // CHECK: %[[W1:.+]] = h %[[W0]]
  // CHECK: return <%[[C]]> %[[W1]] : <i32> !qtx.wire
  qtx.circuit @return_both(%w0 : !qtx.wire) -> <i32>(!qtx.wire) {
    %c = arith.constant 42 : i32
    %w1 = h %w0 : !qtx.wire
    return <%c> %w1 : <i32> !qtx.wire
  }

  // CHECK-LABEL: qtx.circuit @foo(
  // CHECK-SAME:                   %[[VAL_0:.*]]: !qtx.wire,
  // CHECK-SAME:                   %[[VAL_1:.*]]: !qtx.wire_array<2>) -> <i32>(!qtx.wire, !qtx.wire_array<2>) {
  // CHECK:         %[[VAL_2:.*]] = arith.constant 42 : i32
  // CHECK:         return <%[[VAL_2]]> %[[VAL_0]], %[[VAL_1]] : <i32> !qtx.wire, !qtx.wire_array<2>
  // CHECK:       }
  qtx.circuit @foo(%w0: !qtx.wire, %a0: !qtx.wire_array<2>) -> <i32>(!qtx.wire, !qtx.wire_array<2>) {
    %c = arith.constant 42 : i32
    return <%c> %w0, %a0 : <i32> !qtx.wire, !qtx.wire_array<2>
  }

  // CHECK-LABEL: qtx.circuit @rz<
  // CHECK-SAME:                  %[[VAL_0:.*]]: f32>(
  // CHECK-SAME:                  %[[VAL_1:.*]]: !qtx.wire) -> (!qtx.wire) {
  // CHECK:         %[[VAL_2:.*]] = rz<%[[VAL_0]]> %[[VAL_1]] : <f32> !qtx.wire
  // CHECK:         return %[[VAL_2]] : !qtx.wire
  // CHECK:       }
  qtx.circuit @rz<%a: f32>(%w0: !qtx.wire) -> (!qtx.wire) {
    %w1 = rz<%a> %w0 : <f32> !qtx.wire
    return %w1 : !qtx.wire
  }

  // CHECK-LABEL: qtx.circuit @bar() {
  // CHECK:         %[[VAL_0:.*]] = alloca : !qtx.wire
  // CHECK:         %[[VAL_1:.*]] = alloca : !qtx.wire
  // CHECK:         %[[VAL_2:.*]] = alloca : !qtx.wire_array<2>
  // CHECK:         %[[VAL_3:.*]] = arith.constant 3.140000e+00 : f32
  // CHECK:         %[[VAL_4:.*]], %[[VAL_5:.*]]:2 = apply @foo(%[[VAL_1]], %[[VAL_2]]): (!qtx.wire, !qtx.wire_array<2>) -> <i32>(!qtx.wire, !qtx.wire_array<2>)
  // CHECK:         %[[VAL_6:.*]], %[[VAL_7:.*]]:2 = apply<adj> @foo(%[[VAL_5]]#0, %[[VAL_5]]#1): (!qtx.wire, !qtx.wire_array<2>) -> <i32>(!qtx.wire, !qtx.wire_array<2>)
  // CHECK:         %[[VAL_8:.*]], %[[VAL_9:.*]]:2 = apply {{\[}}%[[VAL_0]]] @foo(%[[VAL_7]]#0, %[[VAL_7]]#1): (!qtx.wire, !qtx.wire_array<2>) -> <i32>(!qtx.wire, !qtx.wire_array<2>)
  // CHECK:         %[[VAL_10:.*]] = apply @rz<%[[VAL_3]]>(%[[VAL_0]]): <f32>(!qtx.wire) -> (!qtx.wire)
  // CHECK:         dealloc %[[VAL_10]], %[[VAL_9]]#0, %[[VAL_9]]#1 : !qtx.wire, !qtx.wire, !qtx.wire_array<2>
  // CHECK:         return
  // CHECK:       }
  qtx.circuit @bar() {
    %control = alloca : !qtx.wire
    %w0 = alloca : !qtx.wire
    %a0 = alloca : !qtx.wire_array<2>
    %angle = arith.constant 3.14 : f32
    %cst0, %w1, %a1 = apply @foo(%w0, %a0) : (!qtx.wire, !qtx.wire_array<2>) -> <i32>(!qtx.wire, !qtx.wire_array<2>)
    %cst1, %w2, %a2 = apply<adj> @foo(%w1, %a1) : (!qtx.wire, !qtx.wire_array<2>) -> <i32>(!qtx.wire, !qtx.wire_array<2>)
    %cst2, %w3, %a3 = apply [%control] @foo(%w2, %a2) : (!qtx.wire, !qtx.wire_array<2>) -> <i32>(!qtx.wire, !qtx.wire_array<2>)
    %control_1 = apply @rz<%angle>(%control) : <f32>(!qtx.wire) -> (!qtx.wire)
    dealloc %control_1, %w3, %a3 : !qtx.wire, !qtx.wire, !qtx.wire_array<2>
    return
  }
}
