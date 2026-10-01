// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --split-input-file --verify-diagnostics

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.gadget @work() { fabric.return }
  phys.machine @arch {
    phys.resource_class @q {count = 1 : i64, kind = "qubit",
      native_actions = []}
  }
  phys.graph @graph on @arch : () -> () {
    // expected-error@+1 {{template_event must resolve to exactly one earlier phys.call in the same graph}}
    "phys.call_template"() <{callee = @work, event_id = "call0",
      instance = "root.work.call0", template_event = "later"}> : () -> ()
    "phys.call"() <{callee = @work, event_id = "later",
      instance = "root.work.call1"}> ({ phys.yield }) : () -> ()
    phys.return
  }
}

// -----

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.gadget @work() { fabric.return }
  phys.machine @arch {
    phys.resource_class @q {count = 2 : i64, kind = "qubit",
      native_actions = []}
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.resource @q1 {index = 1 : i64, kind = "qubit", resource_class = @q}
  phys.graph @graph on @arch : () -> () {
    %left = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    %right = phys.acquire [@q1] {event_id = "acquire1"} : !phys.state<@q1>
    %left_out, %right_out = "phys.call"(%left, %right) <{
      callee = @work, event_id = "call0", instance = "root.work.call0"
    }> ({
    ^bb0(%arg0: !phys.state<@q0>, %arg1: !phys.state<@q1>):
      phys.yield %arg0, %arg1 : !phys.state<@q0>, !phys.state<@q1>
    }) : (!phys.state<@q0>, !phys.state<@q1>) ->
         (!phys.state<@q0>, !phys.state<@q1>)
    // expected-error@+1 {{elided state aliases must preserve unique physical owners}}
    "phys.call_template"() <{callee = @work, event_id = "call1",
      instance = "root.work.call1", state_boundary_elided,
      state_aliases = [{alias = @q1, template = @q0}],
      template_event = "call0"}> : () -> ()
    phys.release %left_out {event_id = "release0"} : !phys.state<@q0>
    phys.release %right_out {event_id = "release1"} : !phys.state<@q1>
    phys.return
  }
}

// -----

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.gadget @work() { fabric.return }
  phys.machine @arch {
    phys.resource_class @q {count = 1 : i64, kind = "qubit",
      native_actions = []}
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @graph on @arch : () -> () {
    %state = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    %next = "phys.call"(%state) <{callee = @work, event_id = "call0",
      instance = "root.work.call0"}> ({
    ^bb0(%arg0: !phys.state<@q0>):
      phys.yield %arg0 : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    // expected-error@+1 {{state_boundary_elided requires an empty invocation boundary}}
    %bad = "phys.call_template"(%next) <{callee = @work, event_id = "call1",
      instance = "root.work.call1", state_boundary_elided,
      template_event = "call0"}> : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %bad {event_id = "release0"} : !phys.state<@q0>
    phys.return
  }
}

// -----

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.gadget @work() { fabric.return }
  phys.machine @arch {
    phys.resource_class @q {count = 1 : i64, kind = "qubit",
      native_actions = []}
  }
  phys.graph @graph on @arch : () -> () {
    "phys.call"() <{callee = @work, event_id = "call0",
      instance = "root.work.call0"}> ({ phys.yield }) : () -> ()
    // expected-error@+1 {{state_boundary_elided requires an all-state, type-identical canonical call boundary}}
    "phys.call_template"() <{callee = @work, event_id = "call1",
      instance = "root.work.call1", state_boundary_elided,
      template_event = "call0"}> : () -> ()
    phys.return
  }
}

// -----

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.gadget @work() { fabric.return }
  phys.machine @arch {
    phys.resource_class @q {count = 2 : i64, kind = "qubit",
      native_actions = []}
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.resource @q1 {index = 1 : i64, kind = "qubit", resource_class = @q}
  phys.graph @graph on @arch : () -> () {
    %left = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    %right = phys.acquire [@q1] {event_id = "acquire1"} : !phys.state<@q1>
    %left_out = "phys.call"(%left) <{callee = @work, event_id = "call0",
      instance = "root.work.call0"}> ({
    ^bb0(%state: !phys.state<@q0>):
      phys.yield %state : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    // expected-error@+1 {{state_aliases must exactly cover every renamed physical state resource}}
    %right_out = phys.call_template %right {callee = @work, event_id = "call1",
      instance = "root.work.call1", template_event = "call0"}
      : (!phys.state<@q1>) -> !phys.state<@q1>
    phys.release %left_out {event_id = "release0"} : !phys.state<@q0>
    phys.release %right_out {event_id = "release1"} : !phys.state<@q1>
    phys.return
  }
}

// -----

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.gadget @work() { fabric.return }
  phys.machine @arch {
    phys.resource_class @q {count = 1 : i64, kind = "qubit",
      native_actions = []}
    phys.resource_class @memory {count = 1 : i64, kind = "qubit",
      native_actions = []}
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.resource @m0 {index = 0 : i64, kind = "qubit",
    resource_class = @memory}
  phys.graph @graph on @arch : () -> () {
    %left = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    %right = phys.acquire [@m0] {event_id = "acquire1"} : !phys.state<@m0>
    %left_out = "phys.call"(%left) <{callee = @work, event_id = "call0",
      instance = "root.work.call0"}> ({
    ^bb0(%state: !phys.state<@q0>):
      phys.yield %state : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    // expected-error@+1 {{state aliases must preserve physical resource class and kind}}
    %right_out = phys.call_template %right {callee = @work, event_id = "call1",
      instance = "root.work.call1",
      state_aliases = [{alias = @m0, template = @q0}],
      template_event = "call0"}
      : (!phys.state<@m0>) -> !phys.state<@m0>
    phys.release %left_out {event_id = "release0"} : !phys.state<@q0>
    phys.release %right_out {event_id = "release1"} : !phys.state<@m0>
    phys.return
  }
}

// -----

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.gadget @work() { fabric.return }
  fabric.gadget @other() { fabric.return }
  phys.machine @arch {
    phys.resource_class @q {count = 1 : i64, kind = "qubit",
      native_actions = []}
  }
  phys.graph @graph on @arch : () -> () {
    "phys.call"() <{callee = @work, event_id = "call0",
      instance = "root.work.call0"}> ({ phys.yield }) : () -> ()
    // expected-error@+1 {{a different callee requires a physically equivalent compiler-generated realization}}
    "phys.call_template"() <{callee = @other, event_id = "call1",
      instance = "root.other.call0", template_event = "call0"}> : () -> ()
    phys.return
  }
}

// -----

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.gadget @work() { fabric.return }
  phys.machine @arch {
    phys.resource_class @q {count = 1 : i64, kind = "qubit",
      native_actions = []}
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @graph on @arch : () -> () {
    %state = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    "phys.call"() <{callee = @work, event_id = "call0",
      instance = "root.work.call0"}> ({ phys.yield }) : () -> ()
    // expected-error@+1 {{input and output arities must match the canonical phys.call}}
    %next = "phys.call_template"(%state) <{callee = @work, event_id = "call1",
      instance = "root.work.call1", template_event = "call0"}>
      : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %next {event_id = "release0"} : !phys.state<@q0>
    phys.return
  }
}
