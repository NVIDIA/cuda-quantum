// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --convert-quake-to-qlx %s | FileCheck %s --implicit-check-not=quake. --implicit-check-not=func.func

module {
  qlx.program @existing : () -> () attributes {
    qlx.profile = "p0",
    qlx.stage = "p0"
  } {
    qlx.return
  }

  func.func @__nvqpp__mlirgen__imported()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %wire = quake.null_wire
    %result = quake.h %wire : (!quake.wire) -> !quake.wire
    quake.sink %result : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @existing
// CHECK-LABEL: qlx.program @imported
// CHECK: %[[Q:.*]] = qlx.prepare "zero"
// CHECK: %[[H:.*]] = qlx.apply #qlx.action<h>(%[[Q]])
// CHECK: qlx.discard %[[H]]
