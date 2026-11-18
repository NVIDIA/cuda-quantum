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
