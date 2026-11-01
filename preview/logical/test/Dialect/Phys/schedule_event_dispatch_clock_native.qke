// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=g result=g_schedule' \
// RUN:   | FileCheck %s
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=g result=g_schedule' \
// RUN:   | FileCheck %s
// RUN: env QLX_PROFILE_P2_TO_P3=1 qlx-opt %s \
// RUN:   --phys-schedule='graph=g result=g_schedule' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WORK

module attributes {qlx.profiles = ["p3"], qlx.stages = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @g on @arch : () -> () {
    %q = phys.acquire [@q0] {event_id = "acquire"}
      : !phys.state<@q0>
    %p = phys.prepare %q {event_id = "prepare", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    %r = phys.reset %p {event_id = "reset", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    %event = phys.resource_request "t_state" from @m::@magic {
      event_id = "request"
    } : !event.handle<!phys.resource_payload<@t_state>, "linear">
    "event.try_take"(%event) <{event_id = "take"}> ({
    ^bb0(%resource: !phys.resource_payload<@t_state>):
      phys.discard_resource_payload %resource {event_id = "discard.ready"}
        : !phys.resource_payload<@t_state>
      event.yield
    }, {
    ^bb0(%pending: !event.handle<!phys.resource_payload<@t_state>, "linear">):
      %resource = event.await %pending {event_id = "await.pending"}
        : !event.handle<!phys.resource_payload<@t_state>, "linear">
          -> !phys.resource_payload<@t_state>
      phys.discard_resource_payload %resource {event_id = "discard.pending"}
        : !phys.resource_payload<@t_state>
      event.yield
    }, {
    ^bb0(%status: i8):
      phys.barrier {domains = ["clock"], event_id = "branch_tick"}
        : () -> ()
      event.yield
    }) : (!event.handle<!phys.resource_payload<@t_state>, "linear">) -> ()
    phys.release %r {event_id = "release"} : !phys.state<@q0>
    phys.return
  }
}

// The failed path advances the clock at 2 ns, while the pending path remains
// active until 3 ns.  Their differing clock producers merge through the
// dispatch envelope, so every continuation observes @take at the full 3 ns
// envelope finish.
// CHECK: phys.schedule @g_schedule for @g
// CHECK-SAME: "take|try_take|1|2|control:take
// CHECK-SAME: "discard.pending|discard_resource_payload|2|1|control:discard.pending
// CHECK-SAME: "branch_tick|barrier|2|0|control:branch_tick
// CHECK-SAME: "release|release|3|0|qubits[0]
// CHECK-SAME: resource_deps=reset|domain_deps=take
// CHECK-SAME: makespan_ns = 3.000000e+00 : f64

// Three exclusive dispatch branches touch only five journal keys across both
// independent frontier replays; neither proof copies the pre-existing map.
// WORK: phys-schedule: verifier-frontier state-copies=0 journal-touches=1
