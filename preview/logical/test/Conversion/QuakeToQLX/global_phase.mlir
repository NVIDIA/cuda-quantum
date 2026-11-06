// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --prepare-quake-for-qlx %s | FileCheck %s --check-prefix=CLEAN --implicit-check-not=quake.phase
// RUN: qlx-opt --prepare-quake-for-qlx --convert-quake-to-qlx %s | FileCheck %s --check-prefix=P0 --implicit-check-not=quake. --implicit-check-not=pauli_rotation

// Both ordinary and adjoint global phases forward the anchor, including when
// the successor is used by a gate and then by a measurement.
module {
  func.func @__nvqpp__mlirgen__global_phase() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = arith.constant 0.3 : f64
    %q = quake.null_wire
    %p = quake.phase (%a) %q : (f64, !quake.wire) -> !quake.wire
    %h = quake.h %p : (!quake.wire) -> !quake.wire
    %adj = quake.phase<adj> (%a) %h : (f64, !quake.wire) -> !quake.wire
    %m, %w = quake.mz %adj : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    quake.sink %w : !quake.wire
    return
  }
}

// CLEAN-LABEL: func.func @__nvqpp__mlirgen__global_phase
// CLEAN: %[[Q:.*]] = quake.null_wire
// CLEAN-NEXT: %[[H:.*]] = quake.h %[[Q]]
// CLEAN-NEXT: %{{.*}}, %{{.*}} = quake.mz %[[H]]

// P0-LABEL: qlx.program @global_phase
// P0: %[[Q:.*]] = qlx.prepare "zero"
// P0-NEXT: %[[H:.*]] = qlx.apply #qlx.action<h>(%[[Q]])
// P0-NEXT: %{{.*}} = qlx.measure <Z> %[[H]]
