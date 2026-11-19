// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file %s --qlx-synthesize-rotations | FileCheck %s

/// A Z-axis rotation needs no basis change: T alone realizes it.
qlx.program @rotation_z_quarter_pi : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0.78539816339744828 : f64
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                     sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @rotation_z_quarter_pi
//       CHECK:   %[[Q:.*]] = qlx.prepare
//       CHECK:   qlx.apply #qlx.action<t>(%[[Q]])
/// Every rotation has been synthesized away.
//   CHECK-NOT:   pauli_rotation

// -----

/// An X-axis rotation is conjugated into the Z basis, so the same T is
/// bracketed by the H pair that performs the basis change.
qlx.program @rotation_x_quarter_pi : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0.78539816339744828 : f64
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                     sign = 1 : i64, x_mask = 1 : i64, z_mask = 0 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @rotation_x_quarter_pi
//       CHECK:   %[[Q:.*]] = qlx.prepare
//       CHECK:   %[[H0:.*]] = qlx.apply #qlx.action<h>(%[[Q]])
//       CHECK:   %[[T:.*]] = qlx.apply #qlx.action<t>(%[[H0]])
//       CHECK:   qlx.apply #qlx.action<h>(%[[T]])
//   CHECK-NOT:   pauli_rotation

// -----

/// Nothing to synthesize: a program already in the generator set is left as
/// it is, so the pass is safe to run more than once in a pipeline.
qlx.program @no_rotation_already_clifford_t : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %h = qlx.apply #qlx.action<h>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %t = qlx.apply #qlx.action<t>(%h) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %t : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @no_rotation_already_clifford_t
//       CHECK:   %[[Q:.*]] = qlx.prepare
//       CHECK:   %[[H:.*]] = qlx.apply #qlx.action<h>(%[[Q]])
//       CHECK:   qlx.apply #qlx.action<t>(%[[H]])
