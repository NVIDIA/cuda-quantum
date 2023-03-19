// ========================================================================== //
// Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                 //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: cudaq-opt %s | cudaq-opt | FileCheck %s

module {
  // CHECK-LABEL: @alloc_dealloc
  // CHECK: %[[WIRE:.+]] = alloca : !qtx.wire
  // CHECK: %[[ARRAY:.+]] = alloca : !qtx.wire_array<5>
  // CHECK: %[[ARRAY_D:.+]] = alloca : !qtx.wire_array<2>
  // CHECK: dealloc %[[WIRE]] : !qtx.wire
  // CHECK: dealloc %[[ARRAY]] : !qtx.wire_array<5>
  // CHECK: dealloc %[[ARRAY_D]] : !qtx.wire_array<2>
  qtx.circuit @alloc_dealloc() {
    %wire = alloca : !qtx.wire
    %array = alloca : !qtx.wire_array<5>
    %array_d = alloca : !qtx.wire_array<2, dead = 0>
    dealloc %wire : !qtx.wire
    dealloc %array : !qtx.wire_array<5>
    dealloc %array_d : !qtx.wire_array<2, dead = 0>
    return
  }

  // CHECK-LABEL: @one_target_ops
  // CHECK: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK: %[[W1:.+]] = x %[[W0]] : !qtx.wire
  // CHECK: %[[W2:.+]] = y %[[W1]] : !qtx.wire
  // CHECK: %[[W3:.+]] = z %[[W2]] : !qtx.wire
  // CHECK: %[[W4:.+]] = h %[[W3]] : !qtx.wire
  // CHECK: %[[W5:.+]] = s %[[W4]] : !qtx.wire
  // CHECK: %[[W6:.+]] = t %[[W5]] : !qtx.wire
  // CHECK: dealloc %[[W6]] : !qtx.wire
  qtx.circuit @one_target_ops() {
    %w0 = alloca : !qtx.wire
    %w1 = x %w0 : !qtx.wire
    %w2 = y %w1 : !qtx.wire
    %w3 = z %w2 : !qtx.wire
    %w4 = h %w3 : !qtx.wire
    %w5 = s %w4 : !qtx.wire
    %w6 = t %w5 : !qtx.wire
    dealloc %w6 : !qtx.wire
    return
  }

  // CHECK-LABEL: @controlled_one_target_ops
  // CHECK: %[[CW:.+]] = alloca : !qtx.wire
  // CHECK: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK: %[[W1:.+]] = x [%[[CW]]] %[[W0]] : [!qtx.wire] !qtx.wire
  // CHECK: %[[W2:.+]] = y [%[[CW]]] %[[W1]] : [!qtx.wire] !qtx.wire
  // CHECK: %[[W3:.+]] = z [%[[CW]]] %[[W2]] : [!qtx.wire] !qtx.wire
  // CHECK: %[[W4:.+]] = h [%[[CW]]] %[[W3]] : [!qtx.wire] !qtx.wire
  // CHECK: %[[W5:.+]] = s [%[[CW]]] %[[W4]] : [!qtx.wire] !qtx.wire
  // CHECK: %[[W6:.+]] = t [%[[CW]]] %[[W5]] : [!qtx.wire] !qtx.wire
  // CHECK: dealloc %[[W6]] : !qtx.wire
  // CHECK: dealloc %[[CW]] : !qtx.wire
  qtx.circuit @controlled_one_target_ops() {
    %cw = alloca : !qtx.wire
    %w0 = alloca : !qtx.wire
    %w1 = x [%cw] %w0 : [!qtx.wire] !qtx.wire
    %w2 = y [%cw] %w1 : [!qtx.wire] !qtx.wire
    %w3 = z [%cw] %w2 : [!qtx.wire] !qtx.wire
    %w4 = h [%cw] %w3 : [!qtx.wire] !qtx.wire
    %w5 = s [%cw] %w4 : [!qtx.wire] !qtx.wire
    %w6 = t [%cw] %w5 : [!qtx.wire] !qtx.wire
    dealloc %w6 : !qtx.wire
    dealloc %cw : !qtx.wire
    return
  }

  // CHECK-LABEL: @two_target_ops
  // CHECK: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK: %[[W1:.+]] = alloca : !qtx.wire
  // CHECK: %[[W2:.+]]:2 = swap %[[W0]], %[[W1]] : !qtx.wire, !qtx.wire
  // CHECK: dealloc %[[W2]]#0 : !qtx.wire
  // CHECK: dealloc %[[W2]]#1 : !qtx.wire
  qtx.circuit @two_target_ops() {
    %w0 = alloca : !qtx.wire
    %w1 = alloca : !qtx.wire
    %w2, %w3 = swap %w0, %w1 : !qtx.wire, !qtx.wire
    dealloc %w2 : !qtx.wire
    dealloc %w3 : !qtx.wire
    return
  }

  // CHECK-LABEL: @controlled_two_target_ops
  // CHECK: %[[CW:.+]] = alloca : !qtx.wire
  // CHECK: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK: %[[W1:.+]] = alloca : !qtx.wire
  // CHECK: %[[W2:.+]]:2 = swap [%[[CW]]] %[[W0]], %[[W1]] : [!qtx.wire] !qtx.wire, !qtx.wire
  // CHECK: dealloc %[[W2]]#0 : !qtx.wire
  // CHECK: dealloc %[[W2]]#1 : !qtx.wire
  // CHECK: dealloc %[[CW]] : !qtx.wire
  qtx.circuit @controlled_two_target_ops() {
    %cw = alloca : !qtx.wire
    %w0 = alloca : !qtx.wire
    %w1 = alloca : !qtx.wire
    %w2, %w3 = swap [%cw] %w0, %w1 : [!qtx.wire] !qtx.wire, !qtx.wire
    dealloc %w2 : !qtx.wire
    dealloc %w3 : !qtx.wire
    dealloc %cw : !qtx.wire
    return
  }

  // CHECK-LABEL: @one_target_params_ops
  // CHECK: %[[A64:.+]] = arith.constant
  // CHECK: %[[A32:.+]] = arith.constant
  // CHECK: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK: %[[W1:.+]] = r1<%[[A64]]> %[[W0]] : <f64> !qtx.wire
  // CHECK: %[[W2:.+]] = rx<%[[A64]]> %[[W1]] : <f64> !qtx.wire
  // CHECK: %[[W3:.+]] = ry<%[[A32]]> %[[W2]] : <f32> !qtx.wire
  // CHECK: %[[W4:.+]] = rz<%[[A32]]> %[[W3]] : <f32> !qtx.wire
  // CHECK: dealloc %[[W4]] : !qtx.wire
  qtx.circuit @one_target_params_ops() {
    %a64 = arith.constant 3.14 : f64
    %a32 = arith.constant 3.14 : f32
    %w0 = alloca : !qtx.wire
    %w1 = r1<%a64> %w0 : <f64> !qtx.wire
    %w2 = rx<%a64> %w1 : <f64> !qtx.wire
    %w3 = ry<%a32> %w2 : <f32> !qtx.wire
    %w4 = rz<%a32> %w3 : <f32> !qtx.wire
    dealloc %w4 : !qtx.wire
    return
  }

  // CHECK-LABEL: @controlled_one_target_params_ops
  // CHECK: %[[A64:.+]] = arith.constant
  // CHECK: %[[A32:.+]] = arith.constant
  // CHECK: %[[CW:.+]] = alloca : !qtx.wire
  // CHECK: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK: %[[W1:.+]] = r1<%[[A32]]> [%[[CW]]] %[[W0]] : <f32> [!qtx.wire] !qtx.wire
  // CHECK: %[[W2:.+]] = rx<%[[A32]]> [%[[CW]]] %[[W1]] : <f32> [!qtx.wire] !qtx.wire
  // CHECK: %[[W3:.+]] = ry<%[[A64]]> [%[[CW]]] %[[W2]] : <f64> [!qtx.wire] !qtx.wire
  // CHECK: %[[W4:.+]] = rz<%[[A64]]> [%[[CW]]] %[[W3]] : <f64> [!qtx.wire] !qtx.wire
  // CHECK: dealloc %[[W4]] : !qtx.wire
  // CHECK: dealloc %[[CW]] : !qtx.wire
  qtx.circuit @controlled_one_target_params_ops() {
    %a64 = arith.constant 3.14 : f64
    %a32 = arith.constant 3.14 : f32
    %cw = alloca : !qtx.wire
    %w0 = alloca : !qtx.wire
    %w1 = r1<%a32> [%cw] %w0 : <f32> [!qtx.wire] !qtx.wire
    %w2 = rx<%a32> [%cw] %w1 : <f32> [!qtx.wire] !qtx.wire
    %w3 = ry<%a64> [%cw] %w2 : <f64> [!qtx.wire] !qtx.wire
    %w4 = rz<%a64> [%cw] %w3 : <f64> [!qtx.wire] !qtx.wire
    dealloc %w4 : !qtx.wire
    dealloc %cw : !qtx.wire
    return
  }

  // CHECK-LABEL: @mz_wire
  // CHECK: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK: %[[BIT:.+]], %[[W1:.+]] = mz %[[W0]] : !qtx.wire -> <i1> !qtx.wire
  // CHECK: dealloc %[[W1]] : !qtx.wire
  // CHECK: return <%[[BIT]]> : <i1>
  qtx.circuit @mz_wire() -> <i1> {
    %w0 = alloca : !qtx.wire
    %bit, %w1 = mz %w0 : !qtx.wire -> <i1> !qtx.wire
    dealloc %w1 : !qtx.wire
    return <%bit> : <i1>
  }

  // CHECK-LABEL: @reset_wire
  // CHECK: %[[W0:.+]] = alloca : !qtx.wire
  // CHECK: %[[W1:.+]] = reset %[[W0]] : !qtx.wire
  // CHECK: dealloc %[[W1]] : !qtx.wire
  qtx.circuit @reset_wire() {
    %w0 = alloca : !qtx.wire
    %w1 = reset %w0 : !qtx.wire
    dealloc %w1 : !qtx.wire
    return
  }
}
