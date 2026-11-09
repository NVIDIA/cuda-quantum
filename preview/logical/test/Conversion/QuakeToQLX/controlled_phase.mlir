// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --prepare-quake-for-qlx %s | FileCheck %s
// RUN: not qlx-opt --convert-quake-to-qlx %s 2>&1 | FileCheck %s --check-prefix=ERROR

// A predicate phase is physical and must never be erased as a global phase.
module {
  func.func @__nvqpp__mlirgen__controlled_phase() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = arith.constant 0.3 : f64
    %c = quake.null_wire
    %q = quake.null_wire
    %p:2 = quake.phase (%a) [%c] %q : (f64, !quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
    quake.sink %p#0 : !quake.wire
    quake.sink %p#1 : !quake.wire
    return
  }
}

// CHECK-LABEL: func.func @__nvqpp__mlirgen__controlled_phase
// CHECK: %[[C:.*]] = quake.null_wire
// CHECK: %[[Q:.*]] = quake.null_wire
// CHECK: %[[P:.*]]:2 = quake.phase (%{{.*}}) [%[[C]]] %[[Q]]
// CHECK: quake.sink %[[P]]#0
// CHECK: quake.sink %[[P]]#1
// ERROR: 'quake.phase' op controlled phases are outside the Quake-to-P0 contract; lower upstream
