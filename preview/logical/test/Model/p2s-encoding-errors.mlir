// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  fabric.code @a {distance = 1 : i64, partitions = {data = 1 : i64}}
  fabric.code @b {distance = 1 : i64, partitions = {data = 1 : i64}}
  fabric.code_profile @a_profile {code = @a}
  // expected-error @+1 {{profile and encoding must reference the same code}}
  fabric.encoding @bad {
    block = "block0", code = @b, logical_ports = ["q0"], profile = @a_profile
  }
}

// -----

module {
  // expected-error @+1 {{initial must name a declared phase}}
  fabric.encoding_epoch_schema @bad {
    phases = ["even"], initial = "odd", transitions = []
  }
}

// -----

module {
  fabric.encoding_epoch_schema @schema {
    phases = ["even"], initial = "even", transitions = []
  }
  // expected-error @+1 {{phase must belong to the referenced epoch schema}}
  fabric.encoding_epoch @bad {
    encoding = @external_encoding,
    index = 0 : i64,
    phase = "odd",
    schema = @schema
  }
}

// -----

module {
  fabric.code @a {distance = 1 : i64, partitions = {data = 1 : i64}}
  fabric.code @b {distance = 1 : i64, partitions = {data = 1 : i64}}
  fabric.code_profile @b_profile {code = @b}
  fabric.encoding @b_encoding {
    block = "block0", code = @b, logical_ports = ["q0"], profile = @b_profile
  }
  // expected-error @+1 {{encoding-qualified type code @a disagrees with encoding @b_encoding code @b}}
  fabric.protocol @bad : (!fabric.patch<@a, @b_encoding>) -> !fabric.patch<@a, @b_encoding> {
  ^bb0(%arg0: !fabric.patch<@a, @b_encoding>):
    fabric.protocol_return %arg0 : !fabric.patch<@a, @b_encoding>
  }
}
