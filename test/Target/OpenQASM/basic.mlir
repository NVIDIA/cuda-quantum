// ========================================================================== //
// Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                 //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: cudaq-translate --convert-to=openqasm %s | FileCheck %s

module {
  qtx.circuit @maj(%a0: !qtx.wire, %b0: !qtx.wire, %c0: !qtx.wire) -> (!qtx.wire, !qtx.wire, !qtx.wire) {
    %a1 = x [%c0] %a0 : [!qtx.wire] !qtx.wire
    %b1 = x [%c0] %b0 : [!qtx.wire] !qtx.wire
    %c1 = x [%a1, %b1] %c0 : [!qtx.wire, !qtx.wire] !qtx.wire
    return %a1, %b1, %c1 : !qtx.wire, !qtx.wire, !qtx.wire
  }

  qtx.circuit @umaj(%a0: !qtx.wire, %b0: !qtx.wire, %c0: !qtx.wire) -> (!qtx.wire, !qtx.wire, !qtx.wire) {
    %c1 = x [%a0, %b0] %c0 : [!qtx.wire, !qtx.wire] !qtx.wire
    %b1 = x [%c1] %b0 : [!qtx.wire] !qtx.wire
    %a1 = x [%c1] %a0 : [!qtx.wire] !qtx.wire
    return %a1, %b1, %c1 : !qtx.wire, !qtx.wire, !qtx.wire
  }

  qtx.circuit @ripple_carry_adder() attributes {"cudaq-entrypoint"} {
    %cin = alloca : !qtx.wire
    %a = alloca : !qtx.wire_array<4>
    %b = alloca : !qtx.wire_array<4>
    %cout = alloca : !qtx.wire
    
    // Extract wires:
    %cst0 = arith.constant 0 : index
    %cst1 = arith.constant 1 : index
    %cst2 = arith.constant 2 : index
    %cst3 = arith.constant 3 : index
    %a0, %a1, %a2, %a3, %new_a = array_borrow %cst0, %cst1, %cst2, %cst3 from %a : index, index, index, index from !qtx.wire_array<4> -> !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire_array<4, dead = 4> 
    %b0, %b1, %b2, %b3, %new_b = array_borrow %cst0, %cst1, %cst2, %cst3 from %b : index, index, index, index from !qtx.wire_array<4> -> !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire_array<4, dead = 4> 
    
    // Input states
    %a0_1 = x %a0 : !qtx.wire // a = 0001

    // b = 1111
    %b0_1 = x %b0 : !qtx.wire
    %b1_1 = x %b1 : !qtx.wire
    %b2_1 = x %b2 : !qtx.wire
    %b3_1 = x %b3 : !qtx.wire

    // add a to b, storing result in b majority cin[0],b[0],a[0];
    %cin_1, %b0_2, %a0_2 = apply @maj(%cin, %b0_1, %a0_1) : (!qtx.wire, !qtx.wire, !qtx.wire) -> (!qtx.wire, !qtx.wire, !qtx.wire)

    %a0_3, %b1_2, %a1_1 = apply @maj(%a0_2, %b1_1, %a1) : (!qtx.wire, !qtx.wire, !qtx.wire) -> (!qtx.wire, !qtx.wire, !qtx.wire)
    %a1_2, %b2_2, %a2_1 = apply @maj(%a1_1, %b2_1, %a2) : (!qtx.wire, !qtx.wire, !qtx.wire) -> (!qtx.wire, !qtx.wire, !qtx.wire)
    %a2_2, %b3_2, %a3_1 = apply @maj(%a2_1, %b3_1, %a3) : (!qtx.wire, !qtx.wire, !qtx.wire) -> (!qtx.wire, !qtx.wire, !qtx.wire)

    %cout_1 = x [%a3_1] %cout : [!qtx.wire] !qtx.wire

    %a2_3, %b3_3, %a3_2 = apply @umaj(%a2_2, %b3_2, %a3_1) : (!qtx.wire, !qtx.wire, !qtx.wire) -> (!qtx.wire, !qtx.wire, !qtx.wire)
    %a1_3, %b2_3, %a2_4 = apply @umaj(%a1_2, %b2_2, %a2_3) : (!qtx.wire, !qtx.wire, !qtx.wire) -> (!qtx.wire, !qtx.wire, !qtx.wire)
    %a0_4, %b1_3, %a1_4 = apply @umaj(%a0_3, %b1_2, %a1_3) : (!qtx.wire, !qtx.wire, !qtx.wire) -> (!qtx.wire, !qtx.wire, !qtx.wire)

    %cout_2, %b0_3, %a0_5 = apply @umaj(%cout_1, %b0_2, %a0_4) : (!qtx.wire, !qtx.wire, !qtx.wire) -> (!qtx.wire, !qtx.wire, !qtx.wire)

    %new_a_1 = array_yield %a0_5, %a1_4, %a2_4, %a3_2 to %new_a : !qtx.wire_array<4, dead = 4> -> !qtx.wire_array<4> 
    %new_b_1 = array_yield %b0_3, %b1_3, %b2_3, %b3_3 to %new_b : !qtx.wire_array<4, dead = 4> -> !qtx.wire_array<4> 
    %ans, %new_b_2 = mz %new_b_1 : !qtx.wire_array<4> -> <vector<4xi1>> !qtx.wire_array<4>
    %ans_cout, %cout_3 = mz %cout_2 : !qtx.wire -> <i1> !qtx.wire
    return
  }
}


// CHECK: OPENQASM 2.0;

// CHECK: include "qelib1.inc";

// CHECK: gate maj q0, q1, q2 {
// CHECK:   cx  q2, q0;
// CHECK:   cx  q2, q1;
// CHECK:   ccx  q0, q1, q2;
// CHECK: }

// CHECK: gate umaj q0, q1, q2 {
// CHECK:   ccx  q0, q1, q2;
// CHECK:   cx  q2, q1;
// CHECK:   cx  q2, q0;
// CHECK: }

// CHECK: qreg var0[1];
// CHECK: qreg var1[4];
// CHECK: qreg var2[4];
// CHECK: qreg var3[1];
// CHECK: x  var1[0];
// CHECK: x  var2[0];
// CHECK: x  var2[1];
// CHECK: x  var2[2];
// CHECK: x  var2[3];
// CHECK: maj var0[0], var2[0], var1[0];
// CHECK: maj var1[0], var2[1], var1[1];
// CHECK: maj var1[1], var2[2], var1[2];
// CHECK: maj var1[2], var2[3], var1[3];
// CHECK: cx  var1[3], var3[0];
// CHECK: umaj var1[2], var2[3], var1[3];
// CHECK: umaj var1[1], var2[2], var1[2];
// CHECK: umaj var1[0], var2[1], var1[1];
// CHECK: umaj var3[0], var2[0], var1[0];
// CHECK: creg var12[4];
// CHECK: measure var2 -> var12;
// CHECK: creg var13[1];
// CHECK: measure var3[0] -> var13[0];
