// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module attributes {qlx.profiles = ["p2s", "p2n"]} {
  fabric.code @c {distance = 3 : i64, partitions = {data = 1 : i64}}
  fabric.code_profile @profile {code = @c}
  fabric.encoding_epoch_schema @schema {
    phases = ["even", "odd"],
    initial = "even",
    transitions = ["even->odd", "odd->even"]
  }
  fabric.encoding @encoding {
    block = "block0", code = @c, epoch_schema = @schema,
    initial_epoch = @even, logical_ports = ["q0"], profile = @profile
  }
  fabric.encoding_epoch @even {
    encoding = @encoding, index = 0 : i64, phase = "even", schema = @schema
  }
  fabric.protocol @bad :
      (!fabric.patch<@c, @encoding, @even>) ->
      !fabric.patch<@c, @encoding, @even> {
  ^bb0(%arg0: !fabric.patch<@c, @encoding, @even>):
    // expected-error @+1 {{transition 'even->even' is not declared by epoch schema @schema}}
    %0 = fabric.epoch_transition %arg0 to @even {evidence = "invalid"}
      : (!fabric.patch<@c, @encoding, @even>) ->
        !fabric.patch<@c, @encoding, @even>
    fabric.protocol_return %0 : !fabric.patch<@c, @encoding, @even>
  }
}
