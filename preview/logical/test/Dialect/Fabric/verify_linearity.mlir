// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --qlx-verify-linearity | FileCheck %s

module {
  fabric.code @c {distance = 1 : i64, n = 1 : i64, k = 1 : i64, hx = [],
                  hz = [], lx = [array<i64: 0>], lz = [array<i64: 0>],
                  partitions = {data = 1 : i64}}

  // Exclusive branches each consume the captured patch once.
  // CHECK-LABEL: fabric.gadget @b
  fabric.gadget @b(%arg0: !fabric.patch<@c>, %cond: i1) -> !fabric.patch<@c> {
    %r = cflow.if %cond -> !fabric.patch<@c> {
      %h = fabric.h %arg0 data : !fabric.patch<@c>
      cflow.yield %h : !fabric.patch<@c>
    } else {
      cflow.yield %arg0 : !fabric.patch<@c>
    }
    fabric.return %r : !fabric.patch<@c>
  }

  // event.test observes before event.await consumes.
  // CHECK-LABEL: fabric.gadget @o
  fabric.gadget @o(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    %ev = fabric.resource_request "t_state" from @stream
        : !event.handle<!fabric.resource<@t_state>, "linear">
    %ready = event.test %ev
        : !event.handle<!fabric.resource<@t_state>, "linear"> -> i1
    %res = event.await %ev
        : !event.handle<!fabric.resource<@t_state>, "linear">
        -> !fabric.resource<@t_state>
    fabric.discard_resource %res : !fabric.resource<@t_state>
    fabric.return %arg0 : !fabric.patch<@c>
  }

  // A "shared" handle is not linear: observing it without a consume is fine.
  // CHECK-LABEL: fabric.gadget @shared
  fabric.gadget @shared(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    %ev = fabric.resource_request "t_state" from @stream
        : !event.handle<!fabric.resource<@t_state>, "shared">
    %ready = event.test %ev
        : !event.handle<!fabric.resource<@t_state>, "shared"> -> i1
    fabric.return %arg0 : !fabric.patch<@c>
  }
}
