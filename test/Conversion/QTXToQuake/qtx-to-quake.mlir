// ========================================================================== //
// Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                 //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// TODO: The conversion of QTX to Quake is not fully functional at this time.
// RUN: cudaq-opt --convert-qtx-to-quake %s | FileCheck %s

module {

  // CHECK-LABEL:   func.func @id(
  // CHECK-SAME:                  %[[VAL_0:.*]]: !quake.ref) {
  // CHECK:           return
  // CHECK:         }
  qtx.circuit @id(%q0: !qtx.wire) -> (!qtx.wire) {
    return %q0 : !qtx.wire
  }

  // CHECK-LABEL:   func.func @apply_one_target_operators(
  // CHECK-SAME:                                   %[[VAL_0:.*]]: !quake.ref,
  // CHECK-SAME:                                   %[[VAL_1:.*]]: !quake.ref) {
  // CHECK:   quake.h [%[[VAL_0]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
  // CHECK:   quake.s [%[[VAL_0]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
  // CHECK:   quake.t [%[[VAL_0]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
  // CHECK:   quake.x [%[[VAL_0]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
  // CHECK:   quake.y [%[[VAL_0]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
  // CHECK:   quake.z [%[[VAL_0]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
  // CHECK:           return
  // CHECK:         }
  qtx.circuit @apply_one_target_operators(%q0: !qtx.wire, %q1: !qtx.wire) -> (!qtx.wire, !qtx.wire) {
    %q2 = h [%q0] %q1 : [!qtx.wire] !qtx.wire
    %q3 = s [%q0] %q2 : [!qtx.wire] !qtx.wire
    %q4 = t [%q0] %q3 : [!qtx.wire] !qtx.wire
    %q5 = x [%q0] %q4 : [!qtx.wire] !qtx.wire
    %q6 = y [%q0] %q5 : [!qtx.wire] !qtx.wire
    %q7 = z [%q0] %q6 : [!qtx.wire] !qtx.wire
    return %q0, %q7 : !qtx.wire, !qtx.wire
  }

  // CHECK-LABEL:   func.func @apply_parametrized_one_target_operators() {
  // CHECK:           %[[VAL_0:.*]] = arith.constant 3.140000e+00 : f64
  // CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
  // CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
  // CHECK: quake.r1 (%[[VAL_0]]) [%[[VAL_2]]] %[[VAL_1]] : (f64, !quake.ref, !quake.ref) -> ()
  // CHECK: quake.rx (%[[VAL_0]]) [%[[VAL_2]]] %[[VAL_1]] : (f64, !quake.ref, !quake.ref) -> ()
  // CHECK: quake.ry (%[[VAL_0]]) [%[[VAL_2]]] %[[VAL_1]] : (f64, !quake.ref, !quake.ref) -> ()
  // CHECK: quake.rz (%[[VAL_0]]) [%[[VAL_2]]] %[[VAL_1]] : (f64, !quake.ref, !quake.ref) -> ()
  // CHECK:           quake.dealloc %[[VAL_1]] : !quake.ref
  // CHECK:           quake.dealloc %[[VAL_2]] : !quake.ref
  // CHECK:           return
  // CHECK:         }
  qtx.circuit @apply_parametrized_one_target_operators() {
    %cst = arith.constant 3.140000e+00 : f64
    %0 = alloca : !qtx.wire
    %1 = alloca : !qtx.wire
    %2 = r1<%cst> [%1] %0 : <f64> [!qtx.wire] !qtx.wire
    %3 = rx<%cst> [%1] %2 : <f64> [!qtx.wire] !qtx.wire
    %4 = ry<%cst> [%1] %3 : <f64> [!qtx.wire] !qtx.wire
    %5 = rz<%cst> [%1] %4 : <f64> [!qtx.wire] !qtx.wire
    dealloc %5 : !qtx.wire
    dealloc %1 : !qtx.wire
    return
  }

  // CHECK-LABEL:   func.func @qextract_and_apply_two_targets_operator() {
  // CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
  // CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
  // CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.qvec<2>
  // CHECK: %[[VAL_3:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_0]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK: %[[VAL_4:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_1]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK: quake.swap %[[VAL_3]], %[[VAL_4]] : (!quake.ref, !quake.ref) -> ()
  // CHECK:           quake.dealloc %[[VAL_2]] : !quake.qvec<2>
  // CHECK:           return
  // CHECK:         }
  qtx.circuit @qextract_and_apply_two_targets_operator() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = alloca : !qtx.wire_array<2>
    %wires:2, %new_array = array_borrow %c0_i32, %c1_i32 from %0 : i32, i32 from !qtx.wire_array<2> -> !qtx.wire, !qtx.wire, !qtx.wire_array<2, dead = 2>
    %1:2 = swap %wires#0, %wires#1 : !qtx.wire, !qtx.wire
    %2 = array_yield %1#0, %1#1 to %new_array : !qtx.wire_array<2, dead = 2> -> !qtx.wire_array<2>
    dealloc %2 : !qtx.wire_array<2>
    return
  }

  // CHECK-LABEL:   func.func @qextract_and_apply_controlled_two_targets_operator() {
  // CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
  // CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
  // CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
  // CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.qvec<3>
  // CHECK: %[[VAL_4:.*]] = quake.extract_ref %[[VAL_3]][%[[VAL_0]]] : (!quake.qvec<3>, i32) -> !quake.ref
  // CHECK: %[[VAL_5:.*]] = quake.extract_ref %[[VAL_3]][%[[VAL_1]]] : (!quake.qvec<3>, i32) -> !quake.ref
  // CHECK: %[[VAL_6:.*]] = quake.extract_ref %[[VAL_3]][%[[VAL_2]]] : (!quake.qvec<3>, i32) -> !quake.ref
  // CHECK: quake.swap [%[[VAL_6]]] %[[VAL_4]], %[[VAL_5]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
  // CHECK:           quake.dealloc %[[VAL_3]] : !quake.qvec<3>
  // CHECK:           return
  // CHECK:         }
  qtx.circuit @qextract_and_apply_controlled_two_targets_operator() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = alloca : !qtx.wire_array<3>
    %wires:3, %new_array = array_borrow %c0_i32, %c1_i32, %c2_i32 from %0 : i32, i32, i32 from !qtx.wire_array<3> -> !qtx.wire, !qtx.wire, !qtx.wire, !qtx.wire_array<3, dead = 3>
    %1:2 = swap [%wires#2] %wires#0, %wires#1 : [!qtx.wire] !qtx.wire, !qtx.wire
    %2 = array_yield %1#0, %1#1, %wires#2 to %new_array : !qtx.wire_array<3, dead = 3> -> !qtx.wire_array<3>
    dealloc %2 : !qtx.wire_array<3>
    return
  }

  // CHECK-LABEL:   func.func @return_array(
  // CHECK-SAME:                            %[[VAL_0:.*]]: !quake.qvec<2>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
  // CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
  // CHECK: %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][%[[VAL_1]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK: %[[VAL_4:.*]] = quake.extract_ref %[[VAL_0]][%[[VAL_2]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK:           quake.h %[[VAL_3]] : (!quake.ref) -> ()
  // CHECK:           quake.x %[[VAL_4]] : (!quake.ref) -> ()
  // CHECK:           return
  // CHECK:         }
  qtx.circuit @return_array(%arg0: !qtx.wire_array<2>) -> (!qtx.wire_array<2>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %wires, %new_array = array_borrow %c0_i32 from %arg0 : i32 from !qtx.wire_array<2> -> !qtx.wire, !qtx.wire_array<2, dead = 1>
    %wires_0, %new_array_1 = array_borrow %c1_i32 from %new_array : i32 from !qtx.wire_array<2, dead = 1> -> !qtx.wire, !qtx.wire_array<2, dead = 2>
    %0 = h %wires : !qtx.wire
    %1 = x %wires_0 : !qtx.wire
    %2 = array_yield %1, %0 to %new_array_1 : !qtx.wire_array<2, dead = 2> -> !qtx.wire_array<2>
    return %2 : !qtx.wire_array<2>
  }

  // CHECK-LABEL:   func.func @reset_wires() {
  // CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
  // CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
  // CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.qvec<2>
  // CHECK: %[[VAL_3:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_0]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK: %[[VAL_4:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_1]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK:           quake.reset %[[VAL_3]] : (!quake.ref) -> ()
  // CHECK:           quake.reset %[[VAL_4]] : (!quake.ref) -> ()
  // CHECK:           quake.dealloc %[[VAL_2]] : !quake.qvec<2>
  // CHECK:           return
  // CHECK:         }
  qtx.circuit @reset_wires() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = alloca : !qtx.wire_array<2>
    %wires:2, %new_array = array_borrow %c0_i32, %c1_i32 from %0 : i32, i32 from !qtx.wire_array<2> -> !qtx.wire, !qtx.wire, !qtx.wire_array<2, dead = 2>
    %1 = reset %wires#0 : !qtx.wire
    %2 = reset %wires#1 : !qtx.wire
    %3 = array_yield %1, %2 to %new_array : !qtx.wire_array<2, dead = 2> -> !qtx.wire_array<2>
    dealloc %3 : !qtx.wire_array<2>
    return
  }

  // CHECK-LABEL:   func.func @reset_array() {
  // CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
  // CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
  // CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.qvec<2>
  // CHECK: %[[VAL_3:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_0]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK: %[[VAL_4:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_1]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK:           quake.h %[[VAL_3]] : (!quake.ref) -> ()
  // CHECK:           quake.x %[[VAL_4]]  : (!quake.ref) -> ()
  // CHECK:           quake.reset %[[VAL_2]] : (!quake.qvec<2>) -> ()
  // CHECK: %[[VAL_5:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_0]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK: %[[VAL_6:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_1]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK:           quake.h %[[VAL_5]] : (!quake.ref) -> ()
  // CHECK:           quake.x %[[VAL_6]] : (!quake.ref) -> ()
  // CHECK:           quake.dealloc %[[VAL_2]] : !quake.qvec<2>
  // CHECK:           return
  // CHECK:         }
  qtx.circuit @reset_array() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = alloca : !qtx.wire_array<2>
    %wires:2, %new_array = array_borrow %c0_i32, %c1_i32 from %0 : i32, i32 from !qtx.wire_array<2> -> !qtx.wire, !qtx.wire, !qtx.wire_array<2, dead = 2>
    %1 = h %wires#0 : !qtx.wire
    %2 = x %wires#1 : !qtx.wire
    %3 = array_yield %1, %2 to %new_array : !qtx.wire_array<2, dead = 2> -> !qtx.wire_array<2>
    %4 = reset %3 : !qtx.wire_array<2>
    %wires_0:2, %new_array_1 = array_borrow %c0_i32, %c1_i32 from %4 : i32, i32 from !qtx.wire_array<2> -> !qtx.wire, !qtx.wire, !qtx.wire_array<2, dead = 2>
    %5 = h %wires_0#0 : !qtx.wire
    %6 = x %wires_0#1 : !qtx.wire
    %7 = array_yield %5, %6 to %new_array_1 : !qtx.wire_array<2, dead = 2> -> !qtx.wire_array<2>
    dealloc %7 : !qtx.wire_array<2>
    return
  }

  // CHECK-LABEL:   func.func @mz_wires() {
  // CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
  // CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
  // CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.qvec<2>
  // CHECK: %[[VAL_3:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_0]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK: %[[VAL_4:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_1]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_3]] : (!quake.ref) -> i1
  // CHECK:           %[[VAL_6:.*]] = quake.mz %[[VAL_4]] : (!quake.ref) -> i1
  // CHECK:           quake.dealloc %[[VAL_2]] : !quake.qvec<2>
  // CHECK:           return
  // CHECK:         }
  qtx.circuit @mz_wires() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = alloca : !qtx.wire_array<2>
    %wires:2, %new_array = array_borrow %c0_i32, %c1_i32 from %0 : i32, i32 from !qtx.wire_array<2> -> !qtx.wire, !qtx.wire, !qtx.wire_array<2, dead = 2>
    %bits, %result_targets = mz %wires#0 : !qtx.wire -> <i1> !qtx.wire
    %bits_0, %result_targets_1 = mz %wires#1 : !qtx.wire -> <i1> !qtx.wire
    %1 = array_yield %result_targets, %result_targets_1 to %new_array : !qtx.wire_array<2, dead = 2> -> !qtx.wire_array<2>
    dealloc %1 : !qtx.wire_array<2>
    return
  }

  // CHECK-LABEL:   func.func @mz_array() {
  // CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
  // CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
  // CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.qvec<2>
  // CHECK: %[[VAL_3:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_0]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK: %[[VAL_4:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_1]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK:           quake.h %[[VAL_3]] : (!quake.ref) -> ()
  // CHECK:           quake.x %[[VAL_4]] : (!quake.ref) -> ()
  // CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_2]] : (!quake.qvec<2>) -> !cc.stdvec<i1>
  // CHECK: %[[VAL_6:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_0]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK: %[[VAL_7:.*]] = quake.extract_ref %[[VAL_2]][%[[VAL_1]]] : (!quake.qvec<2>, i32) -> !quake.ref
  // CHECK:           quake.h %[[VAL_6]] : (!quake.ref) -> ()
  // CHECK:           quake.x %[[VAL_7]] : (!quake.ref) -> ()
  // CHECK:           quake.dealloc %[[VAL_2]] : !quake.qvec<2>
  // CHECK:           return
  // CHECK:         }
  qtx.circuit @mz_array() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = alloca : !qtx.wire_array<2>
    %wires:2, %new_array = array_borrow %c0_i32, %c1_i32 from %0 : i32, i32 from !qtx.wire_array<2> -> !qtx.wire, !qtx.wire, !qtx.wire_array<2, dead = 2>
    %1 = h %wires#0 : !qtx.wire
    %2 = x %wires#1 : !qtx.wire
    %3 = array_yield %1, %2 to %new_array : !qtx.wire_array<2, dead = 2> -> !qtx.wire_array<2>
    %bits, %result_targets = mz %3 : !qtx.wire_array<2> -> <vector<2xi1>> !qtx.wire_array<2>
    %wires_0:2, %new_array_1 = array_borrow %c0_i32, %c1_i32 from %result_targets : i32, i32 from !qtx.wire_array<2> -> !qtx.wire, !qtx.wire, !qtx.wire_array<2, dead = 2>
    %4 = h %wires_0#0 : !qtx.wire
    %5 = x %wires_0#1 : !qtx.wire
    %6 = array_yield %4, %5 to %new_array_1 : !qtx.wire_array<2, dead = 2> -> !qtx.wire_array<2>
    dealloc %6 : !qtx.wire_array<2>
    return
  }
}
