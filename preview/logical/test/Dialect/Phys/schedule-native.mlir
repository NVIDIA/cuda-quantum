// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=events result=events_schedule' | FileCheck %s
// RUN: qlx-opt %s --phys-schedule='graph=events result=events_schedule' \
// RUN:   | sed 's/atoms\[0\],atoms\[1\]/atoms[1],atoms[0]/g' \
// RUN:   | qlx-opt | FileCheck %s --check-prefix=REORDER

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @atoms {
      kind = "qubit", count = 2 : i64,
      native_actions = [@cz, @measure_z]
    }
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit",
    resource_class = @atoms}
  phys.resource @q1 {index = 1 : i64, kind = "qubit",
    resource_class = @atoms}
  phys.graph @events on @arch : () ->
      (!phys.record<@bit>, !phys.record<@bit>) {
    %0:2 = phys.acquire [@q0, @q1] {event_id = "acquire0"}
      : !phys.state<@q0>, !phys.state<@q1>
    %1:2 = phys.prepare %0#0, %0#1 {event_id = "prepare", state = "zero"}
      : (!phys.state<@q0>, !phys.state<@q1>) ->
        (!phys.state<@q0>, !phys.state<@q1>)
    %2:2 = phys.apply @cz(%1#0, %1#1) {event_id = "cz"}
      : (!phys.state<@q0>, !phys.state<@q1>) ->
        (!phys.state<@q0>, !phys.state<@q1>)
    %3 = phys.measure @measure_z(%2#0) {
      destructive, event_id = "m0", record_id = "r0"
    } : (!phys.state<@q0>) -> !phys.record<@bit>
    %4 = phys.measure @measure_z(%2#1) {
      destructive, event_id = "m1", record_id = "r1"
    } : (!phys.state<@q1>) -> !phys.record<@bit>
    phys.return %3, %4 : !phys.record<@bit>, !phys.record<@bit>
  }
}

// CHECK: phys.schedule @events_schedule for @events
// CHECK-SAME: "acquire0|acquire|0|0|atoms[0],atoms[1]|deps=|data_deps=|resource_deps=
// CHECK-SAME: "prepare|prepare|0|1|atoms[0],atoms[1]|deps=acquire0|data_deps=acquire0|resource_deps=acquire0
// CHECK-SAME: "cz|apply|1|1|atoms[0],atoms[1]|deps=prepare|data_deps=prepare|resource_deps=prepare
// CHECK-SAME: "m0|measure|2|1|atoms[0]|deps=cz|data_deps=cz|resource_deps=cz
// CHECK-SAME: "m1|measure|2|1|atoms[1]|deps=cz|data_deps=cz|resource_deps=cz
// CHECK-SAME: makespan_ns = 3.000000e+00 : f64
// Resource claims are sets. Dependency fields retain stable semantic order,
// but serialization may list the same physical identities in another order.
// REORDER: phys.schedule @events_schedule for @events
// REORDER-SAME: "acquire0|acquire|0|0|atoms[1],atoms[0]
