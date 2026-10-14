// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=g result=g_schedule' | FileCheck %s
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=g result=g_schedule' | FileCheck %s
// RUN: env QLX_PROFILE_P2_TO_P3=1 qlx-opt %s \
// RUN:   --phys-schedule='graph=g result=g_schedule' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WORK

module attributes {qlx.profiles = ["p3"]} {
  fabric.gadget @work() { fabric.return }
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }

  phys.graph @g on @arch : () -> () {
    %state = phys.acquire [@q0] {event_id = "acquire"}
      : !phys.state<@q0>
    %repeated = "cflow.repeat"(%state) <{
      count = 3 : i64, event_id = "repeat"
    }> ({
    ^bb0(%active: !phys.state<@q0>):
      %step = phys.delay %active {
        duration_ns = 2.0 : f64, event_id = "step"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      %unused = "phys.call"() <{
        callee = @work, event_id = "control.call", instance = "root.work.0"
      }> ({
        %true = arith.constant true
        %tail = phys.condition %true {
          duration_ns = 3.0 : f64, event_id = "control.tail"
        } : i1 -> i1
        phys.yield %tail : i1
      }) : () -> i1
      // One inspectable zero-count template contains a much later event.  It
      // is verified locally but is inactive in the outer repeat summary.
      "cflow.repeat"() <{count = 0 : i64, event_id = "zero"}> ({
        %true = arith.constant true
        %inactive = phys.condition %true {
          duration_ns = 100.0 : f64, event_id = "inactive.tail"
        } : i1 -> i1
        cflow.yield
      }) : () -> ()
      cflow.yield %step : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %repeated {event_id = "release"} : !phys.state<@q0>
    phys.return
  }
}

// The one-iteration resource frontier is 2 ns, but an unused folded control
// call finishes at 3 ns.  The repeat must scale the complete direct-child
// envelope, not only yielded values and changed resources: 3 * 3 ns = 9 ns.
// CHECK: phys.schedule @g_schedule for @g
// CHECK-SAME: "repeat|repeat|0|9|qubits[0]
// CHECK-SAME: repeat_count=3
// CHECK-SAME: "step|delay|0|2|qubits[0]
// CHECK-SAME: parent=repeat|branch=body
// CHECK-SAME: "control.call|call|0|3|control:control.call
// CHECK-SAME: parent=repeat|branch=body
// CHECK-SAME: "control.tail|condition|0|3|control:control.tail
// CHECK-SAME: parent=control.call|branch=body
// CHECK-SAME: "release|release|9|0|qubits[0]
// CHECK-SAME: makespan_ns = 9.000000e+00 : f64

// The hierarchy proof visits each row once even across nested folded calls and
// inactive zero-count templates.  The threaded and serial RUN lines above
// independently check identical schedule serialization.
// WORK: phys-schedule: verifier-hierarchy entries=8 postorder-visits=8
// WORK: phys-schedule: verifier-frontier state-copies=0 journal-touches=2
// WORK: phys-schedule: verifier-work expected-resource-calls=8 expected-resource-types={{[0-9]+}} expected-resource-dedup-comparisons=0 expected-resource-hash-probes={{[1-9][0-9]*}} expected-resource-max-width={{[0-9]+}} data-dependency-derivations=8
// WORK: phys-schedule: portable-proof rows=8 serialized=8 parsed=8 semantic-verifier-runs=1
