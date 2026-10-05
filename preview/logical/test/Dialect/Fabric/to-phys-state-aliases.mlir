// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=p2 device-symbol=device graph-symbol=p2_physical})' | FileCheck %s

module attributes {qlx.profiles = ["p2n"]} {
  lvm.domain @logical {
    lvm.space @compute {capabilities = [], capacity = 2 : i64}
  }
  fabric.code @code {
    distance = 1 : i64,
    partitions = {data = 1 : i64}
  }
  fabric.machine @qec {
    fabric.region @compute {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>
    }
  }
  fabric.protocol @p2 : () -> () {
    %left = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %right = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %left_zero = fabric.prep_z %left : !fabric.patch<@code>
    %right_zero = fabric.prep_z %right : !fabric.patch<@code>
    %left_out = fabric.call @work(%left_zero)
      : (!fabric.patch<@code>) -> !fabric.patch<@code>
    %right_out = fabric.call @work(%right_zero)
      : (!fabric.patch<@code>) -> !fabric.patch<@code>
    fabric.dealloc %left_out : !fabric.patch<@code>
    fabric.dealloc %right_out : !fabric.patch<@code>
    fabric.protocol_return
  }
  fabric.gadget @work(%patch: !fabric.patch<@code>)
      -> !fabric.patch<@code> {
    %prepared = fabric.prep_x %patch : !fabric.patch<@code>
    fabric.return %prepared : !fabric.patch<@code>
  }
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 2 : i64, native_actions = []
    }
    phys.qec_binding @compute_binding {
      qec_region = @qec::@compute, resources = [@qubits]
    }
  }
  qlx.logical_to_qec @logical_to_qec {
    logical = @logical, qec = @qec,
    entries = [{logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @qec_to_physical {
    qec = @qec, physical = @arch,
    entries = [{qec = "compute", binding = "compute_binding",
                resources = ["qubits"]}]
  }
  qlx.device @device {
    logical = @logical, qec = @qec, physical = @arch,
    logical_to_qec = @logical_to_qec,
    qec_to_physical = @qec_to_physical
  }
}

// CHECK: phys.resource @qubits_0_alloc0
// CHECK: phys.resource @qubits_1_alloc1
// CHECK: %[[LEFT:.*]] = "phys.call"
// CHECK-SAME: callee = @work
// CHECK: phys.call_template
// CHECK-SAME: callee = @work
// CHECK-SAME: state_aliases = [{alias = @qubits_1_alloc1, template = @qubits_0_alloc0}]
// CHECK-SAME: state_boundary_elided
// CHECK-SAME: template_event = "call{{[0-9]+}}"
// CHECK-SAME: : () -> ()
