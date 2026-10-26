// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=clock_effect result=clock_schedule' \
// RUN:   | FileCheck %s
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=clock_effect result=clock_schedule' \
// RUN:   | FileCheck %s

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.gadget @work() { fabric.return }
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 2 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @long_lived {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @independent {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }

  phys.graph @clock_effect on @arch : () -> () {
    "phys.call"() <{
      callee = @work, event_id = "canonical", instance = "root.work.0"
    }> ({
      %state = phys.acquire [@long_lived] {event_id = "acquire.long"}
        : !phys.state<@long_lived>
      phys.barrier {domains = ["clock"], event_id = "canonical.tick"}
        : () -> ()
      %done = phys.delay %state {
        duration_ns = 10.0 : f64, event_id = "delay.long"
      } : (!phys.state<@long_lived>) -> !phys.state<@long_lived>
      phys.release %done {event_id = "release.long"}
        : !phys.state<@long_lived>
      phys.yield
    }) : () -> ()

    phys.call_template {
      callee = @work, event_id = "template", instance = "root.work.1",
      template_event = "canonical"
    } : () -> ()

    "phys.call"() <{
      callee = @work, event_id = "independent", instance = "root.work.2"
    }> ({
      %state = phys.acquire [@independent] {event_id = "acquire.independent"}
        : !phys.state<@independent>
      %done = phys.delay %state {
        duration_ns = 1.0 : f64, event_id = "delay.independent"
      } : (!phys.state<@independent>) -> !phys.state<@independent>
      phys.release %done {event_id = "release.independent"}
        : !phys.state<@independent>
      phys.yield
    }) : () -> ()
    phys.return
  }
}

// The canonical clock effect is ready at 0 ns even though its unrelated
// physical-resource tail keeps the envelope alive until 10 ns.  The compact
// invocation starts at 10 ns because it reuses qubits[0], synchronizes both
// concrete resources through its clock effect at 10 ns, and remains alive
// until 20 ns. The independent sibling therefore starts at 10 ns with a domain
// dependency on the compact envelope.
// CHECK: phys.schedule @clock_schedule for @clock_effect
// CHECK-SAME: "canonical|call|0|10|control:canonical
// CHECK-SAME: "template|call_template|10|10|qubits[0],qubits[1]
// CHECK-SAME: "independent|call|10|1|control:independent
// CHECK-SAME: data_deps=|resource_deps=|domain_deps=template
// CHECK-SAME: makespan_ns = 2.000000e+01 : f64
