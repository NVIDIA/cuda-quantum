// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --convert-quake-to-qlx --split-input-file %s | FileCheck %s --check-prefix=P0 --implicit-check-not=quake. --implicit-check-not=pauli_rotation

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
    %post = quake.phase (%a) %w : (f64, !quake.wire) -> !quake.wire
    quake.sink %post : !quake.wire
    return
  }
}

// P0-LABEL: qlx.program @global_phase
// P0: %[[Q:.*]] = qlx.prepare "zero"
// P0-NEXT: %[[H:.*]] = qlx.apply #qlx.action<h>(%[[Q]])
// P0-NEXT: %{{.*}} = qlx.measure <Z> %[[H]]
// P0-NEXT: }

// -----

// Reference-helper provenance must reach the phase successor, so the wire
// wrapped back into the reference is the current slot value.
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

// P0-LABEL: qlx.program @reference_phase
// P0: %[[Q:.*]] = qlx.prepare "zero"
// P0: %[[R:.*]] = qlx.call @phase_helper(%[[Q]])
// P0: qlx.discard %[[R]]
// P0-LABEL: qlx.program @phase_helper
// P0: ^bb0(%[[ARG:.*]]: !qlx.logical_qubit):
// P0-NEXT: %[[H:.*]] = qlx.apply #qlx.action<h>(%[[ARG]])
// P0-NEXT: qlx.return %[[H]]
