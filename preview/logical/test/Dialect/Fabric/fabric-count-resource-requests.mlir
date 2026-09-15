// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt --fabric-count='root=request_batch' %s | FileCheck %s

// A repeated protocol call multiplies both the typed resource-kind demand and
// the demand attributed to its fully qualified logical stream.
fabric.protocol @request_once : () -> !fabric.resource<@t_state> {
  %event = fabric.resource_request "t_state" from @supply::@t_state_stream
      : !event.handle<!fabric.resource<@t_state>, "linear">
  %resource = event.await %event
      : !event.handle<!fabric.resource<@t_state>, "linear">
        -> !fabric.resource<@t_state>
  fabric.protocol_return %resource : !fabric.resource<@t_state>
}

fabric.protocol @request_batch : () -> () {
  cflow.repeat 3 iter() {
    %resource = fabric.call @request_once()
        : () -> !fabric.resource<@t_state>
    fabric.discard_resource %resource : !fabric.resource<@t_state>
    cflow.yield
  }
  fabric.protocol_return
}

// CHECK: fabric.counts = {
// CHECK-SAME: operation_counts = {call = 3 : i64, discard_resource = 3 : i64, event_await = 3 : i64, repeat = 1 : i64, resource_request = 3 : i64}
// CHECK-SAME: resource_requests = [{count = 3 : i64, kind = "t_state", stream = @supply::@t_state_stream}]
