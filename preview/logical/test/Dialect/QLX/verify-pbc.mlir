// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file -verify-diagnostics %s --qlx-verify-pbc | FileCheck %s

/// +pi/4 (a T-class rotation).
qlx.program @positive_quarter_pi : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0.78539816339744828 : f64
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                     sign = 1 : i64, x_mask = 1 : i64, z_mask = 0 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m:2 = qlx.instrument #qlx.instrument<mpp>(%r)
      {parameters = {sign = 1 : i64, x_mask = 1 : i64, z_mask = 0 : i64}}
      : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
  qlx.return %m#1 : i1
}

// CHECK-LABEL: qlx.program @positive_quarter_pi

// -----

/// -pi/4 (a T-dagger-class rotation) is equally well formed.
qlx.program @negative_quarter_pi : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0.78539816339744828 : f64
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                     sign = -1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m:2 = qlx.instrument #qlx.instrument<mpp>(%r)
      {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
  qlx.return %m#1 : i1
}

// CHECK-LABEL: qlx.program @negative_quarter_pi
