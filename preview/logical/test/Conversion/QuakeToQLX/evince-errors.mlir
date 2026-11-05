// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --convert-quake-to-qlx --split-input-file --verify-diagnostics %s

// Explicit output is not compiler lifetime bookkeeping.
module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__explicit_output()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = quake.null_wire
    // expected-error@+1 {{unsupported quantum lifetime logging shape}}
    %logged = quake.evince %a : (!quake.wire) -> !quake.wire
    quake.sink %logged : !quake.wire
    return
  }
}

// -----

// A variadic evince cannot duplicate a live owner.
module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__duplicate_live_owner()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = quake.null_wire
    // expected-error@+1 {{logs a wire with no live or measured P0 owner}}
    %logged:2 = quake.evince %a, %a
        : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire) {compilerGenerated}
    quake.sink %logged#0 : !quake.wire
    quake.sink %logged#1 : !quake.wire
    return
  }
}

// -----

// Measured lifetime handles are linear too.
module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__duplicate_measured_owner()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = quake.null_wire
    %m, %w = quake.mz %a : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    // expected-error@+1 {{logs a wire with no live or measured P0 owner}}
    %logged:2 = quake.evince %w, %w
        : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire) {compilerGenerated}
    quake.sink %logged#0 : !quake.wire
    quake.sink %logged#1 : !quake.wire
    return
  }
}
