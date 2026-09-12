// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p2n"]} {
  qlx.action @produce_t : () -> !fabric.resource<@t_state> {
    kind = "produce_t_state"
  }
  qlx.action @transport_t : (!fabric.resource<@t_state>) -> !fabric.resource<@t_state> {
    kind = "transport_t_state"
  }
  fabric.protocol @factory : () -> !fabric.resource<@t_state> attributes {
    objective = @produce_t
  } {
    %state = fabric.produce_resource {
      region = @factory_region,
      resource = #fabric.resource<T>,
      protocol = #fabric.spec_only<"distill-15to1-T">
    } : !fabric.resource<@t_state>
    fabric.protocol_return %state : !fabric.resource<@t_state>
  }
  fabric.protocol @handoff : (!fabric.resource<@t_state>) -> !fabric.resource<@t_state> attributes {
    objective = @transport_t
  } {
  ^bb0(%state: !fabric.resource<@t_state>):
    %moved = fabric.transport %state from @factory_region to @compute_region {
      protocol = #fabric.spec_only<"encoded-teleport">
    } : !fabric.resource<@t_state> -> !fabric.resource<@t_state>
    fabric.protocol_return %moved : !fabric.resource<@t_state>
  }
  fabric.protocol @custom_factory : () -> !fabric.resource<@cat_state> {
    %state = fabric.produce_resource {
      region = @cat_factory,
      resource_kind = @cat_state,
      protocol = #fabric.spec_only<"cat-cultivation">
    } : !fabric.resource<@cat_state>
    fabric.protocol_return %state : !fabric.resource<@cat_state>
  }
}

// CHECK: qlx.action @produce_t : () -> !fabric.resource<@t_state>
// CHECK: fabric.protocol @factory
// CHECK: fabric.produce_resource
// CHECK-SAME: protocol = #fabric.spec_only<"distill-15to1-T">
// CHECK: fabric.protocol @handoff
// CHECK: fabric.transport %{{.*}} from @factory_region to @compute_region
// CHECK-SAME: protocol = #fabric.spec_only<"encoded-teleport">
// CHECK: fabric.protocol @custom_factory
// CHECK: resource_kind = @cat_state
