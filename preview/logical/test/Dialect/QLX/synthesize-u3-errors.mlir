// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// Licensed under the Apache License, Version 2.0.

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file -verify-diagnostics %s --qlx-synthesize-rotations

func.func @runtime(%q: !qlx.logical_qubit, %angle: f64) -> !qlx.logical_qubit {
  // expected-error@+1 {{Clifford+T synthesis requires a static angle; specialize the program parameter before synthesis}}
  %r = qlx.apply #qlx.action<u3>(%q, %angle, %angle, %angle) : (!qlx.logical_qubit, f64, f64, f64) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}

// -----

func.func @metadata(%q: !qlx.logical_qubit) -> !qlx.logical_qubit {
  %angle = arith.constant 0.3 : f64
  // expected-error@+1 {{u3 angle disagrees with rational-pi metadata}}
  %r = qlx.apply #qlx.action<u3>(%q, %angle, %angle, %angle) {parameters = {theta_pi_numer = 1 : i64, theta_pi_denom = 2 : i64}} : (!qlx.logical_qubit, f64, f64, f64) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}

// -----

func.func @invalid_parameters(%q: !qlx.logical_qubit) -> !qlx.logical_qubit {
  %angle = arith.constant 0.3 : f64
  // expected-error@+1 {{u3 precision must be an f64 attribute}}
  %r = qlx.apply #qlx.action<u3>(%q, %angle, %angle, %angle) {parameters = {precision = "bad"}} : (!qlx.logical_qubit, f64, f64, f64) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}

// -----

func.func @invalid_parameters(%q: !qlx.logical_qubit) -> !qlx.logical_qubit {
  %angle = arith.constant 0.3 : f64
  // expected-error@+1 {{u3 requires canonical rational-pi metadata}}
  %r = qlx.apply #qlx.action<u3>(%q, %angle, %angle, %angle) {parameters = {theta_pi_numer = 1 : i128, theta_pi_denom = 2 : i128}} : (!qlx.logical_qubit, f64, f64, f64) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}
