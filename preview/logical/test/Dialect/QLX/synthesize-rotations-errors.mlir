// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file -verify-diagnostics %s --qlx-synthesize-rotations

/// Rotations the synthesizer refuses to approximate. Both cases fail closed
/// on the rotation itself rather than silently emitting an inexact sequence.

/// The angle is a program parameter, so no finite gate sequence can be
/// chosen at compile time; the program must be specialized first.
qlx.program @bad_dynamic_angle : (f64) -> i1 attributes {qlx.stage = "p0"} {
^bb0(%angle: f64):
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  // expected-error@+1 {{Clifford+T synthesis requires a static angle}}
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                     sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// An authored `precision` overrides the pass option, so it is validated
/// against the same open interval the pass option is.
qlx.program @bad_authored_precision : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0.3 : f64
  // expected-error@+1 {{rotation precision must be finite and in the open interval (0, 1)}}
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                     precision = 2.0 : f64,
                     sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// A NaN angle has no unitary rotation meaning and must not be snapped to the
/// exact-quarter lattice or passed to the numerical synthesizer.
qlx.program @nan_angle : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0x7FF8000000000000 : f64
  // expected-error@+1 {{rotation angle must be finite}}
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// Positive infinity is rejected before any floating-point lattice arithmetic.
qlx.program @positive_infinite_angle : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0x7FF0000000000000 : f64
  // expected-error@+1 {{rotation angle must be finite}}
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// Negative infinity is rejected as non-finite, independently of the
/// canonical nonnegative-magnitude check used for ordinary finite angles.
qlx.program @negative_infinite_angle : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0xFFF0000000000000 : f64
  // expected-error@+1 {{rotation angle must be finite}}
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// Exact metadata is a lossless sidecar for the operand, not a second angle
/// that may silently override it.
qlx.program @conflicting_exact_angle : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0.3 : f64
  // expected-error@+1 {{exact rotation metadata conflicts with the f64 angle operand}}
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                     sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

qlx.program @partial_exact_angle : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0.7853981633974483 : f64
  // expected-error@+1 {{exact rotation metadata requires both angle_pi_numer and angle_pi_denom}}
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {angle_pi_numer = 1 : i64, sign = 1 : i64,
                     x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

qlx.program @zero_exact_denominator : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0.7853981633974483 : f64
  // expected-error@+1 {{exact rotation metadata requires a positive angle_pi_denom}}
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {angle_pi_denom = 0 : i64, angle_pi_numer = 1 : i64,
                     sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

qlx.program @negative_exact_denominator : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0.7853981633974483 : f64
  // expected-error@+1 {{exact rotation metadata requires a positive angle_pi_denom}}
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {angle_pi_denom = -4 : i64, angle_pi_numer = 1 : i64,
                     sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

qlx.program @noncanonical_exact_fraction : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0.7853981633974483 : f64
  // expected-error@+1 {{exact rotation metadata must be a reduced canonical coefficient in [0, 2)}}
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {angle_pi_denom = 8 : i64, angle_pi_numer = 2 : i64,
                     sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}
