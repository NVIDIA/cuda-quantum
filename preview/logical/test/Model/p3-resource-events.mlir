// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @a {
    phys.resource_class @r {kind = "qubit", count = 1 : i64,
                            native_actions = []}
  }
  phys.resource @q {index = 0 : i64, kind = "qubit", resource_class = @r}
  phys.graph @resource_flow on @a :
      (!phys.state<@q>) -> !phys.state<@q> {
  ^bb0(%q: !phys.state<@q>):
    %first = phys.resource_request "t_state" from @m::@magic
      {event_id = "request0"}
      : !event.handle<!phys.resource_payload<@t_state>, "linear">
    %second = phys.resource_request "t_state" from @m::@magic
      {event_id = "request1"}
      : !event.handle<!phys.resource_payload<@t_state>, "linear">
    %ready = event.test %first {event_id = "test2"}
      : !event.handle<!phys.resource_payload<@t_state>, "linear"> -> i1
    %status = event.poll %first {event_id = "poll3"}
      : !event.handle<!phys.resource_payload<@t_state>, "linear"> -> i8
    %pending = event.is %status "pending" {event_id = "is4"} : i8 -> i1
    %which = event.select_ready(%first, %second)
      {event_id = "select5", policy = "fair"}
      : !event.handle<!phys.resource_payload<@t_state>, "linear">,
        !event.handle<!phys.resource_payload<@t_state>, "linear"> -> index
    %cancelled = event.cancel %second
      {event_id = "cancel6", reason = "unused"}
      : !event.handle<!phys.resource_payload<@t_state>, "linear"> -> i8
    %resource = event.await %first {event_id = "await7"}
      : !event.handle<!phys.resource_payload<@t_state>, "linear">
        -> !phys.resource_payload<@t_state>
    event.selection %pending {event_id = "selection8", mode = "require"} : i1
    event.fence {effects = ["event", "resource"], event_id = "fence9"}
    phys.discard_resource_payload %resource {event_id = "discard10"}
      : !phys.resource_payload<@t_state>
    phys.return %q : !phys.state<@q>
  }
}

// CHECK: phys.resource_request "t_state"
// CHECK: event.poll
// CHECK: event.is
// CHECK: event.select_ready
// CHECK: event.cancel
// CHECK: event.await
// CHECK: event.selection
// CHECK: event.fence
// CHECK: phys.discard_resource_payload
