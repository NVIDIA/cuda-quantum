// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --prepare-quake-for-qlx --split-input-file %s | \
// RUN:   FileCheck %s --check-prefix=PREP --implicit-check-not=quake.phase
// RUN: qlx-opt --prepare-quake-for-qlx --convert-quake-to-qlx \
// RUN:   --split-input-file %s | \
// RUN:   FileCheck %s --check-prefix=P0 --implicit-check-not=quake. \
// RUN:   --implicit-check-not=pauli_rotation

// prepare-quake-for-qlx erases ordinary and adjoint global phases and forwards
// the anchor, including when the successor is used by a gate and then by a
// measurement.
module {
  func.func @__nvqpp__mlirgen__global_phase() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = arith.constant 0.3 : f64
    %q = quake.null_wire
    %p = quake.phase (%a) %q : (f64, !quake.wire) -> !quake.wire
    %h = quake.h %p : (!quake.wire) -> !quake.wire
    %adj = quake.phase<adj> (%a) %h : (f64, !quake.wire) -> !quake.wire
    %m, %w = quake.mz %adj : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %post = quake.phase (%a) %w : (f64, !quake.wire) -> !quake.wire
    quake.sink %post : !quake.wire
    return
  }
}

// PREP-LABEL: func.func @__nvqpp__mlirgen__global_phase
// PREP: %[[Q:.*]] = quake.null_wire
// PREP-NEXT: %[[H:.*]] = quake.h %[[Q]]
// PREP-NEXT: %{{.*}}, %[[W:.*]] = quake.mz %[[H]]
// PREP-NEXT: quake.sink %[[W]]

// P0-LABEL: qlx.program @global_phase
// P0: %[[Q:.*]] = qlx.prepare "zero"
// P0-NEXT: %[[H:.*]] = qlx.apply #qlx.action<h>(%[[Q]])
// P0-NEXT: %{{.*}} = qlx.measure <Z> %[[H]]
// P0-NEXT: }

// -----

// A global phase in a reference helper is erased the same way, so the wire
// wrapped back to the reference is the gate's result.
module {
  func.func @__nvqpp__mlirgen__reference_phase() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %q = quake.null_wire
    %r = quake.apply @phase_helper(%q) : (!quake.wire) -> !quake.wire
    quake.sink %r : !quake.wire
    return
  }
  func.func @phase_helper(%ref: !quake.ref) attributes {"cudaq-kernel"} {
    %a = arith.constant 0.3 : f64
    %q = quake.unwrap %ref : (!quake.ref) -> !quake.wire
    %p = quake.phase (%a) %q : (f64, !quake.wire) -> !quake.wire
    %h = quake.h %p : (!quake.wire) -> !quake.wire
    %adj = quake.phase<adj> (%a) %h : (f64, !quake.wire) -> !quake.wire
    quake.wrap %adj to %ref : !quake.wire, !quake.ref
    return
  }
}

// PREP-LABEL: func.func @phase_helper
// PREP: %[[Q:.*]] = quake.unwrap
// PREP-NEXT: %[[H:.*]] = quake.h %[[Q]]
// PREP-NEXT: quake.wrap %[[H]] to

// P0-LABEL: qlx.program @reference_phase
// P0: %[[Q:.*]] = qlx.prepare "zero"
// P0: %[[R:.*]] = qlx.call @phase_helper(%[[Q]])
// P0: qlx.discard %[[R]]
// P0-LABEL: qlx.program @phase_helper
// P0: ^bb0(%[[ARG:.*]]: !qlx.logical_qubit):
// P0-NEXT: %[[H:.*]] = qlx.apply #qlx.action<h>(%[[ARG]])
// P0-NEXT: qlx.return %[[H]]
