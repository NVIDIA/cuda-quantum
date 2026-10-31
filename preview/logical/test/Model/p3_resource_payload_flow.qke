// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  func.func @factory_and_transport(%input: !phys.resource_payload<@t_state>)
      -> (!phys.resource_payload<@t_state>, !phys.resource_payload<@t_state>) {
    %produced = phys.produce_resource @t_state at @factory {
      protocol = #fabric.spec_only<"distill-15to1">,
      event_id = "produce0"
    } : !phys.resource_payload<@t_state>
    %moved = phys.transport_resource %input from @factory to @compute {
      protocol = #fabric.spec_only<"encoded-teleport">,
      event_id = "transport0"
    } : !phys.resource_payload<@t_state> -> !phys.resource_payload<@t_state>
    return %produced, %moved
      : !phys.resource_payload<@t_state>, !phys.resource_payload<@t_state>
  }
}

// CHECK: phys.produce_resource @t_state at @factory
// CHECK-SAME: protocol = #fabric.spec_only<"distill-15to1">
// CHECK: phys.transport_resource %{{.*}} from @factory to @compute
// CHECK-SAME: protocol = #fabric.spec_only<"encoded-teleport">
