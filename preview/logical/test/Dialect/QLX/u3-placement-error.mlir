// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// Licensed under the Apache License, Version 2.0.

// REQUIRES: qlx-opt
// RUN: qlx-opt -verify-diagnostics %s --qlx-to-lvm

func.func @unlegalized(%q: !qlx.logical_qubit, %angle: f64) -> !qlx.logical_qubit {
  // expected-error@+1 {{u3 must be legalized before placement; run cudaq.logical.compiler.synthesize first.}}
  %r = qlx.apply #qlx.action<u3>(%q, %angle, %angle, %angle) : (!qlx.logical_qubit, f64, f64, f64) -> !qlx.logical_qubit
  return %r : !qlx.logical_qubit
}
