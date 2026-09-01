// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | FileCheck %s

// Resource demand is synchronous and remains a first-class P2 fact for static
// counting.
fabric.protocol @request_t : () -> !fabric.resource<@t_state> {
  %resource = fabric.resource_request "t_state" from @t_state_stream
      : !fabric.resource<@t_state>
  fabric.protocol_return %resource : !fabric.resource<@t_state>
}

// CHECK-LABEL: fabric.protocol @request_t : () -> !fabric.resource<@t_state>
// CHECK: %[[RESOURCE:.*]] = fabric.resource_request "t_state" from @t_state_stream : !fabric.resource<@t_state>
// CHECK: fabric.protocol_return %[[RESOURCE]] : !fabric.resource<@t_state>
