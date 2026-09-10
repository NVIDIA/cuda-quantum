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
  phys.graph @try_take on @a :
      (!phys.state<@q>) -> !phys.state<@q> {
  ^bb0(%q: !phys.state<@q>):
    %event = phys.resource_request "t_state" from @m::@magic {
      event_id = "request"
    }
      : !event.handle<!phys.resource_payload<@t_state>, "linear">
    %out = "event.try_take"(%event, %q) <{event_id = "take"}> ({
    ^bb0(%resource: !phys.resource_payload<@t_state>,
         %state: !phys.state<@q>):
      phys.discard_resource_payload %resource {event_id = "discard.ready"}
        : !phys.resource_payload<@t_state>
      event.yield %state : !phys.state<@q>
    }, {
    ^bb0(%pending: !event.handle<!phys.resource_payload<@t_state>, "linear">,
         %state: !phys.state<@q>):
      %resource = event.await %pending {event_id = "await.pending"}
        : !event.handle<!phys.resource_payload<@t_state>, "linear">
          -> !phys.resource_payload<@t_state>
      phys.discard_resource_payload %resource {event_id = "discard.pending"}
        : !phys.resource_payload<@t_state>
      event.yield %state : !phys.state<@q>
    }, {
    ^bb0(%status: i8, %state: !phys.state<@q>):
      event.yield %state : !phys.state<@q>
    }) : (!event.handle<!phys.resource_payload<@t_state>, "linear">,
          !phys.state<@q>) -> !phys.state<@q>
    phys.return %out : !phys.state<@q>
  }
}

// CHECK: event.try_take
// CHECK: phys.discard_resource_payload
// CHECK: event.yield
// CHECK: event.await
// CHECK: phys.discard_resource_payload
// CHECK: event.yield
// CHECK: ^bb0({{.*}}i8
// CHECK: event.yield
