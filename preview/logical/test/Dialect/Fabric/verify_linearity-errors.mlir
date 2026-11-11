// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file -verify-diagnostics %s --qlx-verify-linearity

// qlx-verify-linearity checks every linear-ownership scope, not only P0, and
// reports every violation rather than the first.

module {
  fabric.code @c {distance = 1 : i64, n = 1 : i64, k = 1 : i64, hx = [],
                  hz = [], lx = [array<i64: 0>], lz = [array<i64: 0>],
                  partitions = {data = 1 : i64}}
  fabric.gadget @g(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    // expected-note@+1 {{earlier consume here}}
    %0 = fabric.h %arg0 data : !fabric.patch<@c>
    // expected-error@+2 {{linear value %arg0 : !fabric.patch<@c> (block argument 0 of fabric.gadget) is consumed more than once: by fabric.h and again by fabric.h}}
    // expected-error@+1 {{linear value %1 : !fabric.patch<@c> (result of fabric.h) is never consumed}}
    %1 = fabric.h %arg0 data : !fabric.patch<@c>
    fabric.return %0 : !fabric.patch<@c>
  }
}

// -----

module {
  fabric.code @c {distance = 1 : i64, n = 1 : i64, k = 1 : i64, hx = [],
                  hz = [], lx = [array<i64: 0>], lz = [array<i64: 0>],
                  partitions = {data = 1 : i64}}
  fabric.gadget @l(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    cflow.repeat 3 iter() {
      // expected-note@+3 {{earlier consume here}}
      // expected-error@+2 {{defined outside cflow.repeat is consumed inside its body by fabric.h}}
      // expected-error@+1 {{linear value %0 : !fabric.patch<@c> (result of fabric.h) is never consumed}}
      %h = fabric.h %arg0 data : !fabric.patch<@c>
      cflow.yield
    }
    // expected-error@+1 {{is consumed more than once: by fabric.h and again by fabric.return}}
    fabric.return %arg0 : !fabric.patch<@c>
  }
}

// -----

/// An observation after the consuming await reads a dead handle.
module {
  fabric.code @c {distance = 1 : i64, n = 1 : i64, k = 1 : i64, hx = [],
                  hz = [], lx = [array<i64: 0>], lz = [array<i64: 0>],
                  partitions = {data = 1 : i64}}
  fabric.gadget @s(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    %ev = fabric.resource_request "t_state" from @stream
        : !event.handle<!fabric.resource<@t_state>, "linear">
    // expected-note@+1 {{earlier consume here}}
    %res = event.await %ev
        : !event.handle<!fabric.resource<@t_state>, "linear">
        -> !fabric.resource<@t_state>
    // expected-error@+1 {{is observed by event.test after being consumed by event.await}}
    %ready = event.test %ev
        : !event.handle<!fabric.resource<@t_state>, "linear"> -> i1
    fabric.discard_resource %res : !fabric.resource<@t_state>
    fabric.return %arg0 : !fabric.patch<@c>
  }
}

// -----

/// A leaked allocation and a leaked linear handle.
module {
  fabric.code @c {distance = 1 : i64, n = 1 : i64, k = 1 : i64, hx = [],
                  hz = [], lx = [array<i64: 0>], lz = [array<i64: 0>],
                  partitions = {data = 1 : i64}}
  fabric.gadget @leak(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    // expected-error@+1 {{linear value %0 : !fabric.patch<@c> (result of fabric.alloc) is never consumed}}
    %0 = fabric.alloc {code = @c, region = @r} : !fabric.patch<@c>
    // expected-error@+1 {{(result of fabric.resource_request) is never consumed}}
    %ev = fabric.resource_request "t_state" from @stream
        : !event.handle<!fabric.resource<@t_state>, "linear">
    fabric.return %arg0 : !fabric.patch<@c>
  }
}
