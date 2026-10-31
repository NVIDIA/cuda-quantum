// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module {
  fabric.code @c {distance = 3 : i64, partitions = {data = 1 : i64}}
  fabric.gadget @rpp(%patch: !fabric.patch<@c>) -> !fabric.patch<@c> {
    fabric.return %patch : !fabric.patch<@c>
  } {
    generated_by = @rpp_compiler,
    action_site = @machine::@site0,
    specialization = {
      x_mask = 0 : i64,
      z_mask = 1 : i64,
      sign = 1 : i64,
      angle = 7.8539816339744828E-01 : f64,
      effective_angle = 7.8539816339744828E-01 : f64,
      precision = 1.0000000000000000E-10 : f64,
      rpp_strategy = "t_injection",
      angle_convention = "exp(-i*theta*P/2)"
    }
  }
}

// CHECK: fabric.gadget @rpp
// CHECK: action_site = @machine::@site0
// CHECK-SAME: generated_by = @rpp_compiler
// CHECK-SAME: rpp_strategy = "t_injection"
