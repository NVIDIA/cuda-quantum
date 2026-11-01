// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --convert-quake-to-qlx --split-input-file %s | FileCheck %s --implicit-check-not=quake. --implicit-check-not="arith.constant -"

module {
  func.func @__nvqpp__mlirgen__rotation_axes() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = arith.constant 0.5 : f64
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.null_wire
    %rx = quake.rx (%a) %0 : (f64, !quake.wire) -> !quake.wire
    %ry = quake.ry (%a) %1 : (f64, !quake.wire) -> !quake.wire
    %rz = quake.rz (%a) %2 : (f64, !quake.wire) -> !quake.wire
    quake.sink %rx : !quake.wire
    quake.sink %ry : !quake.wire
    quake.sink %rz : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @rotation_axes
// CHECK: %[[RX_Q:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[RY_Q:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[RZ_Q:.*]] = qlx.prepare "zero" {allocation = 2 : i64
// CHECK: %[[RX_ANGLE:.*]] = arith.constant 5.000000e-01 : f64
// CHECK: %[[RX_RESULT:.*]] = qlx.apply #qlx.action<pauli_rotation>(%[[RX_Q]], %[[RX_ANGLE]]) {parameters = {sign = 1 : i64, x_mask = 1 : i64, z_mask = 0 : i64}}
// CHECK: %[[RY_ANGLE:.*]] = arith.constant 5.000000e-01 : f64
// CHECK: %[[RY_RESULT:.*]] = qlx.apply #qlx.action<pauli_rotation>(%[[RY_Q]], %[[RY_ANGLE]]) {parameters = {sign = 1 : i64, x_mask = 1 : i64, z_mask = 1 : i64}}
// CHECK: %[[RZ_ANGLE:.*]] = arith.constant 5.000000e-01 : f64
// CHECK: %[[RZ_RESULT:.*]] = qlx.apply #qlx.action<pauli_rotation>(%[[RZ_Q]], %[[RZ_ANGLE]]) {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
// CHECK: qlx.discard %[[RX_RESULT]], %[[RY_RESULT]], %[[RZ_RESULT]]

// -----

module {
  func.func @__nvqpp__mlirgen__r1() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = arith.constant 0.25 : f64
    %0 = quake.null_wire
    %1 = quake.null_wire
    %r1 = quake.r1 (%a) %0 : (f64, !quake.wire) -> !quake.wire
    %r1dg = quake.r1<adj> (%a) %1 : (f64, !quake.wire) -> !quake.wire
    quake.sink %r1 : !quake.wire
    quake.sink %r1dg : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @r1
// CHECK: %[[R1_Q:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[R1DG_Q:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[R1_ANGLE:.*]] = arith.constant 2.500000e-01 : f64
// CHECK: %[[R1_RESULT:.*]] = qlx.apply #qlx.action<pauli_rotation>(%[[R1_Q]], %[[R1_ANGLE]]) {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
// CHECK: %[[R1DG_ANGLE:.*]] = arith.constant 2.500000e-01 : f64
// CHECK: %[[R1DG_RESULT:.*]] = qlx.apply #qlx.action<pauli_rotation>(%[[R1DG_Q]], %[[R1DG_ANGLE]]) {parameters = {sign = -1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
// CHECK: qlx.discard %[[R1_RESULT]], %[[R1DG_RESULT]]

// -----

module {
  func.func @__nvqpp__mlirgen__rz_positive() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = arith.constant 0.5 : f64
    %0 = quake.null_wire
    %rz = quake.rz (%a) %0 : (f64, !quake.wire) -> !quake.wire
    quake.sink %rz : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @rz_positive
// CHECK: %[[RZP_Q:.*]] = qlx.prepare "zero"
// CHECK: %[[RZP_ANGLE:.*]] = arith.constant 5.000000e-01 : f64
// CHECK: %[[RZP_RESULT:.*]] = qlx.apply #qlx.action<pauli_rotation>(%[[RZP_Q]], %[[RZP_ANGLE]]) {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
// CHECK: qlx.discard %[[RZP_RESULT]]

// -----

module {
  func.func @__nvqpp__mlirgen__rz_positive_adjoint() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = arith.constant 0.5 : f64
    %0 = quake.null_wire
    %rz = quake.rz<adj> (%a) %0 : (f64, !quake.wire) -> !quake.wire
    quake.sink %rz : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @rz_positive_adjoint
// CHECK: %[[RZPA_Q:.*]] = qlx.prepare "zero"
// CHECK: %[[RZPA_ANGLE:.*]] = arith.constant 5.000000e-01 : f64
// CHECK: %[[RZPA_RESULT:.*]] = qlx.apply #qlx.action<pauli_rotation>(%[[RZPA_Q]], %[[RZPA_ANGLE]]) {parameters = {sign = -1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
// CHECK: qlx.discard %[[RZPA_RESULT]]

// -----

module {
  func.func @__nvqpp__mlirgen__rz_negative() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = arith.constant -0.5 : f64
    %0 = quake.null_wire
    %rz = quake.rz (%a) %0 : (f64, !quake.wire) -> !quake.wire
    quake.sink %rz : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @rz_negative
// CHECK: %[[RZN_Q:.*]] = qlx.prepare "zero"
// CHECK: %[[RZN_ANGLE:.*]] = arith.constant 5.000000e-01 : f64
// CHECK: %[[RZN_RESULT:.*]] = qlx.apply #qlx.action<pauli_rotation>(%[[RZN_Q]], %[[RZN_ANGLE]]) {parameters = {sign = -1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
// CHECK: qlx.discard %[[RZN_RESULT]]

// -----

module {
  func.func @__nvqpp__mlirgen__rz_negative_adjoint() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = arith.constant -0.5 : f64
    %0 = quake.null_wire
    %rz = quake.rz<adj> (%a) %0 : (f64, !quake.wire) -> !quake.wire
    quake.sink %rz : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @rz_negative_adjoint
// CHECK: %[[RZNA_Q:.*]] = qlx.prepare "zero"
// CHECK: %[[RZNA_ANGLE:.*]] = arith.constant 5.000000e-01 : f64
// CHECK: %[[RZNA_RESULT:.*]] = qlx.apply #qlx.action<pauli_rotation>(%[[RZNA_Q]], %[[RZNA_ANGLE]]) {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
// CHECK: qlx.discard %[[RZNA_RESULT]]
