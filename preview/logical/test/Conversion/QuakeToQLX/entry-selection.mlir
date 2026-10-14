// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt '--convert-quake-to-qlx=entry-point=foo..0x2' %s | FileCheck %s --check-prefix=EXACT
// RUN: not qlx-opt '--convert-quake-to-qlx=entry-point=foo' %s 2>&1 | FileCheck %s --check-prefix=AMBIGUOUS

module {
  func.func @__nvqpp__mlirgen__foo..0x1()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %wire = quake.null_wire
    quake.sink %wire : !quake.wire
    return
  }
  func.func @__nvqpp__mlirgen__foo..0x2()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %wire = quake.null_wire
    %result = quake.h %wire : (!quake.wire) -> !quake.wire
    quake.sink %result : !quake.wire
    return
  }
}

// EXACT-LABEL: qlx.program @foo
// EXACT: qlx.apply #qlx.action<h>
// EXACT-NOT: func.func
// AMBIGUOUS: error: convert-quake-to-qlx entry-point 'foo' identified 2 source entries
