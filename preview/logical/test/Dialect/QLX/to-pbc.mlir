// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file %s --qlx-to-pbc | FileCheck %s

/// No Clifford to absorb: the T becomes a Z-axis rotation and the Z
/// measurement keeps its basis (x_mask = 0, z_mask = 1 for both).
qlx.program @t_without_clifford : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %t = qlx.apply #qlx.action<t>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %t : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @t_without_clifford
//       CHECK:   qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:     x_mask = 0 : i64, z_mask = 1 : i64
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
//  CHECK-SAME:     x_mask = 0 : i64, z_mask = 1 : i64

// -----

/// The H is absorbed rather than emitted: conjugating by it turns both the
/// rotation and the measurement into their X-axis counterparts.
qlx.program @t_after_clifford : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %h = qlx.apply #qlx.action<h>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %t = qlx.apply #qlx.action<t>(%h) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %t : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @t_after_clifford
//       CHECK:   qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:     x_mask = 1 : i64, z_mask = 0 : i64
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
//  CHECK-SAME:     x_mask = 1 : i64, z_mask = 0 : i64
/// PBC form admits no gate actions, so the H and T are gone.
//   CHECK-NOT:   #qlx.action<h>
//   CHECK-NOT:   #qlx.action<t>
