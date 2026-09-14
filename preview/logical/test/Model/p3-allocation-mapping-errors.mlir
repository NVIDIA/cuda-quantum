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
    phys.resource_class @q {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.graph @graph on @arch : () -> () {
    %0 = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    phys.release %0 {event_id = "release0"} : !phys.state<@q0>
    phys.return
  }
  // expected-error @+1 {{acquire event 'missing' must resolve to phys.acquire}}
  phys.allocation_mapping @bad for @graph {
    entries = [{allocation = "scratch", resource_class = @q,
      resources = [@q0], indices = array<i64: 0>, acquire = "missing",
      release = "release0"}]
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @q {
      kind = "qubit", count = 2 : i64, native_actions = []
    }
  }
  phys.resource @first {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.resource @second {kind = "qubit", resource_class = @q, index = 1 : i64}
  phys.graph @graph on @arch : () -> () {
    %0 = phys.acquire [@first] {event_id = "acquire0"}
      : !phys.state<@first>
    %1 = phys.acquire [@second] {event_id = "acquire1"}
      : !phys.state<@second>
    phys.release %0 {event_id = "release0"} : !phys.state<@first>
    phys.release %1 {event_id = "release1"} : !phys.state<@second>
    phys.return
  }
  // expected-error @+1 {{must cover every graph-acquired phys.resource}}
  phys.allocation_mapping @incomplete for @graph {
    entries = [
      {allocation = "first", resource_class = @q, resources = [@first],
       indices = array<i64: 0>, acquire = "acquire0", release = "release0"}
    ]
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @q {
      kind = "qubit", count = 2 : i64, native_actions = []
    }
  }
  phys.resource @first {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.resource @second {kind = "qubit", resource_class = @q, index = 1 : i64}
  phys.graph @graph on @arch : () -> () {
    %0 = phys.acquire [@first] {event_id = "acquire0"}
      : !phys.state<@first>
    %1 = phys.acquire [@second] {event_id = "acquire1"}
      : !phys.state<@second>
    phys.release %0 {event_id = "release0"} : !phys.state<@first>
    phys.release %1 {event_id = "release1"} : !phys.state<@second>
    phys.return
  }
  // expected-error @+1 {{graph must have exactly one phys.allocation_mapping}}
  phys.allocation_mapping @first_half for @graph {
    entries = [
      {allocation = "first", resource_class = @q, resources = [@first],
       indices = array<i64: 0>, acquire = "acquire0", release = "release0"}
    ]
  }
  phys.allocation_mapping @second_half for @graph {
    entries = [
      {allocation = "second", resource_class = @q, resources = [@second],
       indices = array<i64: 1>, acquire = "acquire1", release = "release1"}
    ]
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @q {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @first {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.resource @second {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.graph @graph on @arch : () -> () {
    %0 = phys.acquire [@first] {event_id = "acquire0"}
      : !phys.state<@first>
    %1 = phys.acquire [@second] {event_id = "acquire1"}
      : !phys.state<@second>
    phys.release %0 {event_id = "release0"} : !phys.state<@first>
    phys.release %1 {event_id = "release1"} : !phys.state<@second>
    phys.return
  }
  // expected-error @+1 {{physical resource identity @q[0] has overlapping allocation lifetimes ('first' and 'second')}}
  phys.allocation_mapping @bad for @graph {
    entries = [
      {allocation = "first", resource_class = @q, resources = [@first],
       indices = array<i64: 0>, acquire = "acquire0", release = "release0"},
      {allocation = "second", resource_class = @q, resources = [@second],
       indices = array<i64: 0>, acquire = "acquire1", release = "release1"}
    ]
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @q {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @first {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.resource @second {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.graph @graph on @arch : () -> () {
    %0 = phys.acquire [@first] {event_id = "acquire0"}
      : !phys.state<@first>
    %1 = phys.acquire [@second] {event_id = "acquire1"}
      : !phys.state<@second>
    phys.release %1 {event_id = "release1"} : !phys.state<@second>
    phys.return
  }
  // expected-error @+1 {{physical resource identity @q[0] has overlapping allocation lifetimes ('open' and 'second')}}
  phys.allocation_mapping @bad for @graph {
    entries = [
      {allocation = "open", resource_class = @q, resources = [@first],
       indices = array<i64: 0>, acquire = "acquire0"},
      {allocation = "second", resource_class = @q, resources = [@second],
       indices = array<i64: 0>, acquire = "acquire1", release = "release1"}
    ]
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @q {
      kind = "qubit", count = 2 : i64, native_actions = []
    }
  }
  phys.resource @q0 {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.resource @q1 {kind = "qubit", resource_class = @q, index = 1 : i64}
  phys.graph @graph on @arch : () -> () {
    %0 = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    phys.release %0 {event_id = "release0"} : !phys.state<@q0>
    phys.return
  }
  // expected-error @+1 {{acquire event 'acquire0' resources do not match the allocation binding}}
  phys.allocation_mapping @bad for @graph {
    entries = [{allocation = "scratch", resource_class = @q,
      resources = [@q1], indices = array<i64: 1>, acquire = "acquire0"}]
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @q {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q1 {kind = "qubit", resource_class = @q, index = 1 : i64}
  phys.graph @graph on @arch : () -> () {
    %0 = phys.acquire [@q1] {event_id = "acquire0"} : !phys.state<@q1>
    phys.release %0 {event_id = "release0"} : !phys.state<@q1>
    phys.return
  }
  // expected-error @+1 {{resource @q1 index is outside resource-class capacity}}
  phys.allocation_mapping @bad for @graph {
    entries = [{allocation = "scratch", resource_class = @q,
      resources = [@q1], indices = array<i64: 1>, acquire = "acquire0",
      release = "release0"}]
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @q {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.graph @graph on @arch : () -> () {
    %0 = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    phys.release %0 {event_id = "release0"} : !phys.state<@q0>
    %1 = phys.acquire [@q0] {event_id = "acquire1"} : !phys.state<@q0>
    phys.release %1 {event_id = "release1"} : !phys.state<@q0>
    phys.return
  }
  // expected-error @+1 {{physical resource identity @q[0] has causally unordered adjacent allocation lifetimes}}
  phys.allocation_mapping @bad for @graph {
    entries = [
      {allocation = "first", resource_class = @q, resources = [@q0],
       indices = array<i64: 0>, acquire = "acquire0", release = "release0"},
      {allocation = "second", resource_class = @q, resources = [@q0],
       indices = array<i64: 0>, acquire = "acquire1", release = "release1"}
    ]
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @q {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.graph @graph on @arch : () -> () {
    %0 = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    %1 = phys.acquire [@q0] {event_id = "acquire1"} : !phys.state<@q0>
    phys.release %0 {event_id = "release0"} : !phys.state<@q0>
    phys.release %1 {event_id = "release1"} : !phys.state<@q0>
    phys.return
  }
  // expected-error @+1 {{must cover every graph-acquired phys.resource}}
  phys.allocation_mapping @incomplete for @graph {
    entries = [
      {allocation = "first", resource_class = @q, resources = [@q0],
       indices = array<i64: 0>, acquire = "acquire0", release = "release0"}
    ]
  }
}
