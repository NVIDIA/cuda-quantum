// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  func.func @same(%p: !fabric.patch<@c>) {
    // expected-error @+1 {{source and destination spaces must differ}}
    %0 = fabric.relocate %p using @handoff from @machine::@memory
      to @machine::@memory {
        transition = "move", continuity_witness = "w", step = 0 : i64
      } : !fabric.patch<@c>
    return
  }
}

// -----

module {
  func.func @bad_step(%p: !fabric.patch<@c>) {
    // expected-error @+1 {{step must be nonnegative}}
    %0 = fabric.relocate %p using @handoff from @machine::@memory
      to @machine::@compute {
        transition = "move", continuity_witness = "w", step = -1 : i64
      } : !fabric.patch<@c>
    return
  }
}
