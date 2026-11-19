// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --convert-quake-to-qlx --split-input-file --verify-diagnostics %s

// A global phase must be erased by prepare-quake-for-qlx first.
module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = arith.constant 0.3 : f64
    %q = quake.null_wire
    // expected-error@+1 {{global phase must be erased by prepare-quake-for-qlx}}
    %p = quake.phase (%a) %q : (f64, !quake.wire) -> !quake.wire
    quake.sink %p : !quake.wire
    return
  }
}

// -----

// P0 measurement is destructive, so a helper cannot return a measured qubit
// through its reference.
module {
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %q = quake.null_wire
    %r = quake.apply @helper(%q) : (!quake.wire) -> !quake.wire
    quake.sink %r : !quake.wire
    return
  }
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @helper(%ref: !quake.ref) attributes {"cudaq-kernel"} {
    %q = quake.unwrap %ref : (!quake.ref) -> !quake.wire
    %m, %w = quake.mz %q : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    // expected-error@+1 {{returns a qubit that is measured or not live; P0 measurement is destructive}}
    quake.wrap %w to %ref : !quake.wire, !quake.ref
    return
  }
}

// -----

// A helper-local qubit must be measured or sunk before the helper returns.
module {
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %q = quake.null_wire
    %r = quake.apply @helper(%q) : (!quake.wire) -> !quake.wire
    quake.sink %r : !quake.wire
    return
  }
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @helper(%ref: !quake.ref) attributes {"cudaq-kernel"} {
    %local = quake.null_wire
    %q = quake.unwrap %ref : (!quake.ref) -> !quake.wire
    %c:2 = quake.x [%q] %local : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
    quake.wrap %c#0 to %ref : !quake.wire, !quake.ref
    // expected-error@+1 {{helper leaves 1 live quantum value(s); wrap, measure, or sink every wire}}
    return
  }
}

// -----

// A vector index must fall inside the width the call site specializes.
module {
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = quake.null_wire
    %b = quake.null_wire
    %c = quake.bundle_cable %a, %b : (!quake.wire, !quake.wire) -> !quake.cable<2>
    %r = quake.apply @helper(%c) : (!quake.cable<2>) -> !quake.cable<2>
    %s:2 = quake.split_cable %r : (!quake.cable<2>) -> (!quake.wire, !quake.wire)
    quake.sink %s#0 : !quake.wire
    quake.sink %s#1 : !quake.wire
    return
  }
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @helper(%v: !quake.veq<?>) attributes {"cudaq-kernel"} {
    // expected-error@+1 {{static vector index exceeds the specialized cable width}}
    %r = quake.extract_ref %v[2] : (!quake.veq<?>) -> !quake.ref
    %q = quake.unwrap %r : (!quake.ref) -> !quake.wire
    quake.wrap %q to %r : !quake.wire, !quake.ref
    return
  }
}
