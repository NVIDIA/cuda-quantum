// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module {
  fabric.code @c {distance = 3 : i64, partitions = {data = 1 : i64}}
  // expected-error @+1 {{RPP specialization requires canonical angle_convention}}
  fabric.gadget @bad(%patch: !fabric.patch<@c>) -> !fabric.patch<@c> {
    fabric.return %patch : !fabric.patch<@c>
  } {
    generated_by = @rpp_compiler,
    action_site = @machine::@site0,
    specialization = {
      angle = 1.250000E-01 : f64,
      effective_angle = 1.250000E-01 : f64,
      rpp_strategy = "native",
      angle_convention = "turns"
    }
  }
}
