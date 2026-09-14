// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: not qlx-opt %s --phys-schedule='graph=graph result=schedule' 2>&1 \
// RUN:   | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @graph on @arch : () -> () {
    %state = phys.acquire [@q0] {event_id = "acquire,bad"}
      : !phys.state<@q0>
    phys.release %state {event_id = "release"} : !phys.state<@q0>
    phys.return
  }
}

// Schedule rows use comma-separated list fields.  Fail closed before emitting
// an ambiguous portable schedule instead of silently changing one event into
// several dependency tokens during replay.
// CHECK: schedule event 'acquire,bad' field event_id contains reserved list delimiter '|' or ','
