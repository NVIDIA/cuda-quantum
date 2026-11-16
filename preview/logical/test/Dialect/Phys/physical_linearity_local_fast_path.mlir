// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --split-input-file --verify-diagnostics

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64,
      kind = "qubit",
      native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64,
    kind = "qubit",
    resource_class = @qubits
  }
  phys.graph @single_local_owner on @arch : () -> () {
    %0 = phys.acquire [@q0] {event_id = "acquire"} : !phys.state<@q0>
    %1 = phys.prepare %0 {event_id = "prepare", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %1 {event_id = "release"} : !phys.state<@q0>
    phys.return
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64,
      kind = "qubit",
      native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64,
    kind = "qubit",
    resource_class = @qubits
  }
  // expected-error @+1 {{linear_value_count is 1 but the verified graph defines 2 linear physical values}}
  phys.graph @wrong_linear_count on @arch : () -> () attributes {
    linear_value_count = 1 : i64
  } {
    %0 = phys.acquire [@q0] {event_id = "acquire"} : !phys.state<@q0>
    %1 = phys.prepare %0 {event_id = "prepare", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %1 {event_id = "release"} : !phys.state<@q0>
    phys.return
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64,
      kind = "qubit",
      native_actions = []
    }
  }
  // expected-error @+1 {{physical-state resource @missing must resolve to phys.resource}}
  phys.graph @missing_resource on @arch : () -> () {
    %0 = phys.acquire [@missing] {event_id = "acquire"} : !phys.state<@missing>
    phys.release %0 {event_id = "release"} : !phys.state<@missing>
    phys.return
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64,
      kind = "qubit",
      native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64,
    kind = "qubit",
    resource_class = @qubits
  }
  // expected-error @+1 {{physical state is consumed more than once by phys.prepare and phys.prepare}}
  phys.graph @multiple_local_owners on @arch : () -> () {
    %0 = phys.acquire [@q0] {event_id = "acquire"} : !phys.state<@q0>
    %1 = phys.prepare %0 {event_id = "prepare.zero", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    %2 = phys.prepare %0 {event_id = "prepare.plus", state = "plus"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %1 {event_id = "release.zero"} : !phys.state<@q0>
    phys.release %2 {event_id = "release.plus"} : !phys.state<@q0>
    phys.return
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64,
      kind = "qubit",
      native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64,
    kind = "qubit",
    resource_class = @qubits
  }
  // expected-error @+1 {{physical state captured by conditional control must be consumed on every branch}}
  phys.graph @conditional_owner on @arch : () -> () {
    %0 = phys.acquire [@q0] {event_id = "acquire"} : !phys.state<@q0>
    %condition = arith.constant true
    "cflow.if"(%condition) <{event_id = "if"}> ({
      phys.release %0 {event_id = "release"} : !phys.state<@q0>
      cflow.yield
    }, {
      cflow.yield
    }) : (i1) -> ()
    phys.return
  }
}
