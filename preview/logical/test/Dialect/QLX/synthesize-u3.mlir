// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// Licensed under the Apache License, Version 2.0.

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file %s --qlx-synthesize-rotations --qlx-verify-clifford-t | FileCheck %s

func.func @exact_h(%q: !qlx.logical_qubit) -> !qlx.logical_qubit {
  %theta = arith.constant 1.5707963267948966 : f64
  %phi = arith.constant 0.0 : f64
  %lambda = arith.constant 3.141592653589793 : f64
  %r = qlx.apply #qlx.action<u3>(%q, %theta, %phi, %lambda) {parameters = {theta_pi_numer = 1 : i64, theta_pi_denom = 2 : i64, lambda_pi_numer = 1 : i64, lambda_pi_denom = 1 : i64}} : (!qlx.logical_qubit, f64, f64, f64) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}
// CHECK-LABEL: func.func @exact_h
// CHECK: %[[R:.*]] = qlx.apply #qlx.action<h>
// CHECK-NOT: qlx.apply
// CHECK: return %[[R]]

// -----

func.func @approximate(%q: !qlx.logical_qubit) -> !qlx.logical_qubit {
  %theta = arith.constant 0.3 : f64
  %phi = arith.constant 1.1 : f64
  %lambda = arith.constant 0.4 : f64
  %r = qlx.apply #qlx.action<u3>(%q, %theta, %phi, %lambda) : (!qlx.logical_qubit, f64, f64, f64) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}
// CHECK-LABEL: func.func @approximate
// CHECK-NOT: #qlx.action<u3>
// CHECK: qlx.apply #qlx.action<
// CHECK-NOT: pauli_rotation
// CHECK: return
