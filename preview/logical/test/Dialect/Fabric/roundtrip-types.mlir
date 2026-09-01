// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | qlx-opt | FileCheck %s

// Test all Fabric types round-trip through parsing and printing.

// CHECK-LABEL: func.func @test_patch_type
// CHECK-SAME: %arg0: !fabric.patch<@surface_17>
// CHECK-SAME: -> !fabric.patch<@surface_17>
func.func @test_patch_type(%p: !fabric.patch<@surface_17>) -> !fabric.patch<@surface_17> {
  return %p : !fabric.patch<@surface_17>
}

// CHECK-LABEL: func.func @test_syndrome_type
// CHECK-SAME: %arg0: !fabric.syndrome<@steane>
// CHECK-SAME: -> !fabric.syndrome<@steane>
func.func @test_syndrome_type(%s: !fabric.syndrome<@steane>) -> !fabric.syndrome<@steane> {
  return %s : !fabric.syndrome<@steane>
}

// CHECK-LABEL: func.func @test_syndrome_different_code
// CHECK-SAME: %arg0: !fabric.syndrome<@surface_17>
func.func @test_syndrome_different_code(%s: !fabric.syndrome<@surface_17>) -> !fabric.syndrome<@surface_17> {
  return %s : !fabric.syndrome<@surface_17>
}

// CHECK-LABEL: func.func @test_bit_type
// CHECK-SAME: %arg0: !fabric.bit
func.func @test_bit_type(%b: !fabric.bit) -> !fabric.bit {
  return %b : !fabric.bit
}

// CHECK-LABEL: func.func @test_magic_types
// CHECK-SAME: %arg0: !fabric.resource<T>
// CHECK-SAME: %arg1: !fabric.resource<CCZ>
// CHECK-SAME: %arg2: !fabric.resource<CS>
func.func @test_magic_types(%t: !fabric.resource<T>,
                            %ccz: !fabric.resource<CCZ>,
                            %cs: !fabric.resource<CS>)
    -> (!fabric.resource<T>, !fabric.resource<CCZ>, !fabric.resource<CS>) {
  return %t, %ccz, %cs : !fabric.resource<T>, !fabric.resource<CCZ>, !fabric.resource<CS>
}

// CHECK-LABEL: func.func @test_slot_type
// CHECK-SAME: %arg0: !fabric.slot
func.func @test_slot_type(%s: !fabric.slot) -> !fabric.slot {
  return %s : !fabric.slot
}

// CHECK-LABEL: func.func @test_machine_type
// CHECK-SAME: %arg0: !fabric.machine
func.func @test_machine_type(%machine: !fabric.machine) -> !fabric.machine {
  return %machine : !fabric.machine
}
