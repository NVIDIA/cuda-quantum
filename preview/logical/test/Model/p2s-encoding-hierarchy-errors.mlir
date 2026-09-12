// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  fabric.code @outer {distance = 1 : i64, k = 1 : i64, n = 1 : i64,
    partitions = {data = 1 : i64}, r = 0 : i64}
  fabric.code @child {distance = 1 : i64, k = 2 : i64, n = 2 : i64,
    partitions = {data = 2 : i64}, r = 0 : i64}
  fabric.code @composite {distance = 0 : i64, k = 1 : i64, n = 2 : i64,
    partitions = {data = 2 : i64}, r = 0 : i64}
  fabric.code_profile @outer_profile {code = @outer}
  fabric.code_profile @child_profile {code = @child}
  fabric.code_profile @composite_profile {code = @composite}
  fabric.encoding @outer_encoding {
    block = "block0", code = @outer, logical_ports = ["q0"],
    profile = @outer_profile}
  fabric.encoding @child_encoding {
    block = "block0", code = @child, logical_ports = ["q0", "q1"],
    profile = @child_profile}
  fabric.encoding @flat {
    block = "block0", code = @composite, logical_ports = ["q0"],
    profile = @composite_profile}
  // expected-error @+1 {{every child logical port must be mapped, exposed, gauged, or fixed}}
  fabric.encoding_hierarchy @incomplete for @composite {
    carrier_map = ["0:0:0"], child = @child_encoding, depth = 2 : i64,
    flat_encoding = @flat, multiplicity = 1 : i64,
    outer = @outer_encoding}
}

// -----

module {
  fabric.code @outer {distance = 1 : i64, k = 1 : i64, n = 1 : i64,
    partitions = {data = 1 : i64}, r = 0 : i64}
  fabric.code @child {distance = 1 : i64, k = 2 : i64, n = 2 : i64,
    partitions = {data = 2 : i64}, r = 0 : i64}
  fabric.code @composite {distance = 0 : i64, k = 1 : i64, n = 2 : i64,
    partitions = {data = 2 : i64}, r = 0 : i64}
  fabric.code_profile @outer_profile {code = @outer}
  fabric.code_profile @child_profile {code = @child}
  fabric.code_profile @composite_profile {code = @composite}
  fabric.encoding @outer_encoding {
    block = "block0", code = @outer, logical_ports = ["q0"],
    profile = @outer_profile}
  fabric.encoding @child_encoding {
    block = "block0", code = @child, logical_ports = ["q0", "q1"],
    profile = @child_profile}
  fabric.encoding @flat {
    block = "block0", code = @composite, logical_ports = ["q0"],
    profile = @composite_profile}
  // expected-error @+1 {{fixed_ports entries require valid child, port, x/z basis, +/-1 eigenvalue, and nonempty evidence}}
  fabric.encoding_hierarchy @bad_fixed for @composite {
    carrier_map = ["0:0:0"], child = @child_encoding, depth = 2 : i64,
    fixed_ports = [{basis = "z", child = 0 : i64, eigenvalue = 1 : i64,
                    evidence = "", port = 1 : i64}],
    flat_encoding = @flat, multiplicity = 1 : i64,
    outer = @outer_encoding}
}

// -----

module {
  fabric.code @outer {distance = 1 : i64, k = 1 : i64, n = 1 : i64,
    partitions = {data = 1 : i64}, r = 0 : i64}
  fabric.code @child {distance = 1 : i64, k = 1 : i64, n = 2 : i64,
    partitions = {data = 2 : i64}, r = 1 : i64}
  fabric.code @composite {distance = 0 : i64, k = 1 : i64, n = 2 : i64,
    partitions = {data = 2 : i64}, r = 0 : i64}
  fabric.code_profile @outer_profile {code = @outer}
  fabric.code_profile @child_profile {code = @child}
  fabric.code_profile @composite_profile {code = @composite}
  fabric.encoding @outer_encoding {
    block = "block0", code = @outer, logical_ports = ["q0"],
    profile = @outer_profile}
  fabric.encoding @child_encoding {
    block = "block0", code = @child, logical_ports = ["q0"],
    profile = @child_profile}
  fabric.encoding @flat {
    block = "block0", code = @composite, logical_ports = ["q0"],
    profile = @composite_profile}
  // expected-error @+1 {{composite r must include outer, child, and reclassified gauges}}
  fabric.encoding_hierarchy @lost_gauge for @composite {
    carrier_map = ["0:0:0"], child = @child_encoding, depth = 2 : i64,
    flat_encoding = @flat, multiplicity = 1 : i64,
    outer = @outer_encoding}
}
