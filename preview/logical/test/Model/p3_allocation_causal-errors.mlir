// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: not qlx-opt %s 2>&1 | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @q {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }

  phys.resource @first {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.resource @second {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.graph @textual_only on @arch : () -> () {
    %0 = phys.acquire [@first] {event_id = "acquire0"}
      : !phys.state<@first>
    phys.release %0 {event_id = "release0"} : !phys.state<@first>
    %1 = phys.acquire [@second] {event_id = "acquire1"}
      : !phys.state<@second>
    phys.release %1 {event_id = "release1"} : !phys.state<@second>
    phys.return
  }
  phys.allocation_mapping @textual_only_allocations for @textual_only {
    entries = [
      {allocation = "first", resource_class = @q,
       resources = [@first], indices = array<i64: 0>,
       acquire = "acquire0", release = "release0"},
      {allocation = "second", resource_class = @q,
       resources = [@second], indices = array<i64: 0>,
       acquire = "acquire1", release = "release1"}
    ]
  }
}

// CHECK: physical resource identity @q[0] has causally unordered adjacent allocation lifetimes
