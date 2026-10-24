// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// Licensed under the Apache License, Version 2.0.

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file -verify-diagnostics %s

func.func @valid(%q: !qlx.logical_qubit, %a: f64) -> !qlx.logical_qubit {
  %r = qlx.apply #qlx.action<u3>(%q, %a, %a, %a) {parameters = {precision = 1.0e-6 : f64}} : (!qlx.logical_qubit, f64, f64, f64) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}

// -----

func.func @missing_angle(%q: !qlx.logical_qubit, %a: f64) -> !qlx.logical_qubit {
  // expected-error@+1 {{u3 requires one logical-qubit input and result and three trailing f64 angles}}
  %r = qlx.apply #qlx.action<u3>(%q, %a, %a) : (!qlx.logical_qubit, f64, f64) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}

// -----

func.func @extra_angle(%q: !qlx.logical_qubit, %a: f64) -> !qlx.logical_qubit {
  // expected-error@+1 {{u3 requires one logical-qubit input and result and three trailing f64 angles}}
  %r = qlx.apply #qlx.action<u3>(%q, %a, %a, %a, %a) : (!qlx.logical_qubit, f64, f64, f64, f64) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}

// -----

func.func @extra_qubit(%q: !qlx.logical_qubit, %a: f64) -> !qlx.logical_qubit {
  // expected-error@+1 {{u3 requires one logical-qubit input and result and three trailing f64 angles}}
  %r = qlx.apply #qlx.action<u3>(%q, %q, %a, %a, %a) : (!qlx.logical_qubit, !qlx.logical_qubit, f64, f64, f64) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}

// -----

func.func @non_f64(%q: !qlx.logical_qubit, %a: f64, %i: i64) -> !qlx.logical_qubit {
  // expected-error@+1 {{u3 requires three trailing f64 angle operands}}
  %r = qlx.apply #qlx.action<u3>(%q, %i, %a, %a) : (!qlx.logical_qubit, i64, f64, f64) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}

// -----

func.func @wrong_position(%q: !qlx.logical_qubit, %a: f64) -> !qlx.logical_qubit {
  // expected-error@+1 {{u3 requires one logical-qubit input and result and three trailing f64 angles}}
  %r = qlx.apply #qlx.action<u3>(%a, %q, %a, %a) : (f64, !qlx.logical_qubit, f64, f64) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}

// -----

func.func @resource_u3(%q: !qlx.logical_qubit, %resource: !qlx.logical_resource<"t_state">) -> !qlx.logical_qubit {
  // expected-error@+1 {{resource consumption cannot encode a parameterized u3}}
  %r = qlx.consume_resource %resource with #qlx.action<u3>(%q) : !qlx.logical_resource<"t_state">, (!qlx.logical_qubit) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}
