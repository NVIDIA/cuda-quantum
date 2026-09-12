// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  // expected-error @+1 {{resource kind must match the event payload kind}}
  %0 = phys.resource_request "t_state" from @magic
    : !event.handle<!phys.resource_payload<@ccz_state>, "linear">
}

// -----

module {
  %0 = phys.resource_request "t_state" from @magic
    : !event.handle<!phys.resource_payload<@t_state>, "linear">
  // expected-error @+1 {{failed to verify that result type equals the event payload type}}
  %1 = event.await %0
    : !event.handle<!phys.resource_payload<@t_state>, "linear">
      -> !phys.resource_payload<@ccz_state>
}

// -----

module {
  %0 = phys.resource_request "t_state" from @magic
    : !event.handle<!phys.resource_payload<@t_state>, "linear">
  %1 = phys.resource_request "ccz_state" from @magic
    : !event.handle<!phys.resource_payload<@ccz_state>, "linear">
  // expected-error @+1 {{requires all operands to have the same type}}
  %2 = event.select_ready(%0, %1)
    : !event.handle<!phys.resource_payload<@t_state>, "linear">,
      !event.handle<!phys.resource_payload<@ccz_state>, "linear"> -> index
}

// -----

module {
  %status = arith.constant 0 : i8
  // expected-error @+1 {{event state must be pending, ready, failed, cancelled, or exhausted}}
  %0 = event.is %status "unknown" : i8 -> i1
}

// -----

module {
  // expected-error @+1 {{semantic effects must be unique}}
  event.fence {effects = ["event", "event"]}
}

// -----

module {
  %predicate = arith.constant true
  // expected-error @+1 {{accept_when disagrees with the selection mode}}
  event.selection %predicate {accept_when = false, mode = "require"} : i1
}

// -----

module {
  phys.machine @a {
    phys.resource_class @r {kind = "qubit", count = 1 : i64,
                            native_actions = []}
  }
  phys.graph @bad_try_take on @a :
      (!phys.state<@q>) -> !phys.state<@q> {
  ^bb0(%state: !phys.state<@q>):
    %event = phys.resource_request "t_state" from @magic
      : !event.handle<!phys.resource_payload<@t_state>, "linear">
    // expected-error @+1 {{failed alternative argument has the wrong type}}
    %out = "event.try_take"(%event, %state) ({
    ^bb0(%resource: !phys.resource_payload<@t_state>, %carry: !phys.state<@q>):
      event.yield %carry : !phys.state<@q>
    }, {
    ^bb0(%pending: !event.handle<!phys.resource_payload<@t_state>, "linear">,
         %carry: !phys.state<@q>):
      event.yield %carry : !phys.state<@q>
    }, {
    ^bb0(%wrong: i1, %carry: !phys.state<@q>):
      event.yield %carry : !phys.state<@q>
    }) : (!event.handle<!phys.resource_payload<@t_state>, "linear">,
          !phys.state<@q>) -> !phys.state<@q>
    phys.return %out : !phys.state<@q>
  }
}

// -----

// `phys.state` implements `UniqueCarryOwnerInterface`, so `event.try_take`
// rejects two carries naming the same physical resource. The graph boundary
// itself carries only one `!phys.state<@q>` (the duplicate is constructed by
// passing that same value twice as carries) so this exercises
// `event.try_take`'s own check, not `phys.graph`'s unrelated
// duplicate-boundary-type check.
module {
  phys.machine @a {
    phys.resource_class @r {kind = "qubit", count = 1 : i64,
                            native_actions = []}
  }
  phys.graph @duplicate_carry on @a :
      (!phys.state<@q>) -> !phys.state<@q> {
  ^bb0(%q: !phys.state<@q>):
    %event = phys.resource_request "t_state" from @magic
      : !event.handle<!phys.resource_payload<@t_state>, "linear">
    // expected-error @+1 {{cannot carry more than one owner for resource @q}}
    %out:2 = "event.try_take"(%event, %q, %q) ({
    ^bb0(%resource: !phys.resource_payload<@t_state>,
         %carry0: !phys.state<@q>, %carry1: !phys.state<@q>):
      event.yield %carry0, %carry1 : !phys.state<@q>, !phys.state<@q>
    }, {
    ^bb0(%pending: !event.handle<!phys.resource_payload<@t_state>, "linear">,
         %carry0: !phys.state<@q>, %carry1: !phys.state<@q>):
      event.yield %carry0, %carry1 : !phys.state<@q>, !phys.state<@q>
    }, {
    ^bb0(%status: i8, %carry0: !phys.state<@q>, %carry1: !phys.state<@q>):
      event.yield %carry0, %carry1 : !phys.state<@q>, !phys.state<@q>
    }) : (!event.handle<!phys.resource_payload<@t_state>, "linear">,
          !phys.state<@q>, !phys.state<@q>) -> (!phys.state<@q>, !phys.state<@q>)
    phys.release %out#1 : !phys.state<@q>
    phys.return %out#0 : !phys.state<@q>
  }
}
