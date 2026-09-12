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
    lvm.space @compute {capabilities = [], capacity = 4 : i64}
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
    %left0 = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %right0 = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %left1 = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %right1 = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %left0_zero = fabric.prep_z %left0 : !fabric.patch<@code>
    %right0_zero = fabric.prep_z %right0 : !fabric.patch<@code>
    %left1_zero = fabric.prep_z %left1 : !fabric.patch<@code>
    %right1_zero = fabric.prep_z %right1 : !fabric.patch<@code>
    %left0_out, %right0_out =
      fabric.call @touch_left(%left0_zero, %right0_zero)
        : (!fabric.patch<@code>, !fabric.patch<@code>)
          -> (!fabric.patch<@code>, !fabric.patch<@code>)
    %left1_out, %right1_out =
      fabric.call @touch_left(%left1_zero, %right1_zero)
        : (!fabric.patch<@code>, !fabric.patch<@code>)
          -> (!fabric.patch<@code>, !fabric.patch<@code>)
    fabric.dealloc %left0_out : !fabric.patch<@code>
    fabric.dealloc %right0_out : !fabric.patch<@code>
    fabric.dealloc %left1_out : !fabric.patch<@code>
    fabric.dealloc %right1_out : !fabric.patch<@code>
    fabric.protocol_return
  }
  fabric.gadget @touch_left(%left: !fabric.patch<@code>,
                            %right: !fabric.patch<@code>)
      -> (!fabric.patch<@code>, !fabric.patch<@code>) {
    %prepared = fabric.prep_x %left : !fabric.patch<@code>
    fabric.return %prepared, %right
      : !fabric.patch<@code>, !fabric.patch<@code>
  }
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 4 : i64, native_actions = []
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

// The second patch is returned unchanged by the source gadget.  It therefore
// remains owned by the caller and is absent from both physical call boundaries.
// CHECK: %[[CALL:[^ ]+]] = "phys.call"(%[[LEFT0:[^)]+]]) <{
// CHECK-SAME: callee = @touch_left
// CHECK-SAME: }> ({
// CHECK: ^bb0(%[[BODY_LEFT:.*]]: !phys.state<@qubits_0_alloc0>):
// CHECK: phys.yield %{{.*}} : !phys.state<@qubits_0_alloc0>
// The canonical call's complete state boundary is type-identical, so the
// authenticated alias map is sufficient to reconstruct it without SSA state.
// CHECK: phys.call_template
// CHECK-SAME: state_aliases = [{alias = @qubits_2_alloc2, template = @qubits_0_alloc0}]
// CHECK-SAME: state_boundary_elided
// CHECK-SAME: : () -> ()
// CHECK: phys.release %{{.*}} {event_id = "release{{[0-9]+}}"} : !phys.state<@qubits_3_alloc3>
