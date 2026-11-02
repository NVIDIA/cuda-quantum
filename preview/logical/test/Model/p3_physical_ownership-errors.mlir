// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64,
      kind = "qubit",
      native_actions = [@h]
    }
  }
  phys.resource @q0 {
    index = 0 : i64,
    kind = "qubit",
    resource_class = @qubits
  }
  phys.resource @q1 {
    index = 0 : i64,
    kind = "qubit",
    resource_class = @qubits
  }
  phys.graph @mismatched_acquire on @arch : () -> () {
    // expected-error @+1 {{each acquired state result must be positionally qualified by its matching resource}}
    %0 = phys.acquire [@q0] : !phys.state<@q1>
    phys.release %0 : !phys.state<@q1>
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
  phys.resource @first {
    index = 0 : i64,
    kind = "qubit",
    resource_class = @qubits
  }
  phys.resource @second {
    index = 0 : i64,
    kind = "qubit",
    resource_class = @qubits
  }
  phys.graph @unknown_resource_predecessor on @arch : () -> () {
    %0 = phys.acquire [@first] {event_id = "acquire0"}
      : !phys.state<@first>
    phys.release %0 {event_id = "release0"} : !phys.state<@first>
    %1 = phys.acquire [@second] {event_id = "acquire1"}
      : !phys.state<@second>
    phys.release %1 {event_id = "release1"} : !phys.state<@second>
    phys.return
  }
  // expected-error @+1 {{after release event 'missing' must resolve to a mapped physical lifetime terminator}}
  phys.allocation_mapping @unknown_resource_predecessor_allocations
      for @unknown_resource_predecessor {
    entries = [
      {
        acquire = "acquire0",
        allocation = "first",
        indices = array<i64: 0>,
        release = "release0",
        resource_class = @qubits,
        resources = [@first]
      },
      {
        acquire = "acquire1",
        after = ["missing"],
        allocation = "second",
        indices = array<i64: 0>,
        release = "release1",
        resource_class = @qubits,
        resources = [@second]
      }
    ]
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
  phys.graph @move_drops_owner on @arch : () -> () {
    %0 = phys.acquire [@q0] : !phys.state<@q0>
    // expected-error @+1 {{move must preserve every physical state type}}
    phys.move @route(%0) : (!phys.state<@q0>) -> ()
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
    %0 = phys.acquire [@missing] : !phys.state<@missing>
    phys.release %0 : !phys.state<@missing>
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
  phys.graph @prepare_drops_owner on @arch : () -> () {
    %0 = phys.acquire [@q0] : !phys.state<@q0>
    // expected-error @+1 {{prepare must preserve every physical state type}}
    phys.prepare %0 {state = "zero"} : (!phys.state<@q0>) -> ()
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
  // expected-error @+1 {{physical state crosses unsupported region control scf.if}}
  phys.graph @scf_conditional_release on @arch : () -> () {
    %0 = phys.acquire [@q0] : !phys.state<@q0>
    %condition = arith.constant true
    "scf.if"(%condition) ({
      phys.release %0 : !phys.state<@q0>
      "scf.yield"() : () -> ()
    }, {
      "scf.yield"() : () -> ()
    }) : (i1) -> ()
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
  // expected-error @+1 {{physical state crosses unsupported region control scf.for}}
  phys.graph @scf_repeated_release on @arch : () -> () {
    %0 = phys.acquire [@q0] : !phys.state<@q0>
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    "scf.for"(%lb, %ub, %step) ({
    ^bb0(%iv: index):
      phys.release %0 : !phys.state<@q0>
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
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
  phys.graph @duplicate_repeat_owners on @arch :
      (!phys.state<@q0>) -> !phys.state<@q0> {
  ^bb0(%unrelated: !phys.state<@q0>):
    %0 = phys.acquire [@q0] : !phys.state<@q0>
    // expected-error @+1 {{cannot carry more than one physical-state owner for resource @q0}}
    %1:2 = "cflow.repeat"(%0, %unrelated) <{count = 1 : i64}> ({
    ^bb0(%acquired: !phys.state<@q0>, %other: !phys.state<@q0>):
      cflow.yield %other, %acquired
        : !phys.state<@q0>, !phys.state<@q0>
    }) : (!phys.state<@q0>, !phys.state<@q0>) ->
         (!phys.state<@q0>, !phys.state<@q0>)
    phys.release %1#0 : !phys.state<@q0>
    phys.return %1#1 : !phys.state<@q0>
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
  phys.graph @double_consume on @arch : () -> () {
    %0 = phys.acquire [@q0] : !phys.state<@q0>
    %1 = phys.prepare %0 {state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    %2 = phys.prepare %0 {state = "plus"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %1 : !phys.state<@q0>
    phys.release %2 : !phys.state<@q0>
    phys.return
  }
}

// -----

module {
  phys.action @h {
    arity = 1 : i64,
    process = "{\22kind\22:\22unitary\22,\22matrix\22:[[1,0],[0,1]]}"
  }
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 2 : i64,
      kind = "qubit",
      native_actions = [@h]
    }
  }
  phys.resource @q0 {
    index = 0 : i64,
    kind = "qubit",
    resource_class = @qubits
  }
  phys.resource @q1 {
    index = 1 : i64,
    kind = "qubit",
    resource_class = @qubits
  }
  phys.graph @forged_lineage on @arch :
      (!phys.state<@q0>) -> !phys.state<@q0> {
  ^bb0(%unrelated: !phys.state<@q0>):
    %0:2 = phys.acquire [@q0, @q1] {event_id = "acquire0"}
      : !phys.state<@q0>, !phys.state<@q1>
    %1 = phys.apply @h(%0#1) {event_id = "apply0"}
      : (!phys.state<@q1>) -> !phys.state<@q1>
    phys.release %unrelated, %1 {event_id = "release0"}
      : !phys.state<@q0>, !phys.state<@q1>
    phys.return %0#0 : !phys.state<@q0>
  }
  // expected-error @+1 {{release event 'release0' state for @q0 is not derived from acquire event 'acquire0'}}
  phys.allocation_mapping @forged_lineage_allocations for @forged_lineage {
    entries = [{
      acquire = "acquire0",
      allocation = "data",
      indices = array<i64: 0, 1>,
      release = "release0",
      resource_class = @qubits,
      resources = [@q0, @q1]
    }]
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
  phys.graph @conditional_release on @arch : () -> () {
    %0 = phys.acquire [@q0] : !phys.state<@q0>
    %condition = arith.constant true
    "cflow.if"(%condition) ({
      phys.release %0 : !phys.state<@q0>
      cflow.yield
    }, {
      cflow.yield
    }) : (i1) -> ()
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
  // expected-error @+1 {{physical state defined outside a loop is consumed inside its body}}
  phys.graph @loop_capture_release on @arch : () -> () {
    %0 = phys.acquire [@q0] : !phys.state<@q0>
    "cflow.repeat"() <{count = 2 : i64}> ({
      phys.release %0 : !phys.state<@q0>
      cflow.yield
    }) : () -> ()
    phys.return
  }
}
