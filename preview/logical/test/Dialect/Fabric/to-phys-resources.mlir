// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=p2 device-symbol=device})' | FileCheck %s

module attributes {qlx.profiles = ["p2n"]} {
  lvm.domain @logical {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
    lvm.stream @magic {produces = @t_state, capacity = 1 : i64}
  }
  fabric.code @code {
    distance = 1 : i64, k = 1 : i64, n = 1 : i64, r = 0 : i64,
    partitions = {data = 1 : i64},
    lx = [array<i64: 0>], lz = [array<i64: 0>]
  }
  fabric.machine @qec {
    fabric.region @compute {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>
    }
  }
  fabric.protocol @p2 : () -> () {
    %patch = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %event = fabric.resource_request "t_state" from @logical::@magic
      : !event.handle<!fabric.resource<@t_state>, "linear">
    %resource = event.await %event
      : !event.handle<!fabric.resource<@t_state>, "linear">
        -> !fabric.resource<@t_state>
    %rotated = fabric.resource_rotate_product %resource on %patch {
      patch_indices = array<i64: 0>, logical_indices = array<i64: 0>,
      pauli_product = "-Z", angle = 7.8539816339744828e-01 : f64
    } : (!fabric.resource<@t_state>, !fabric.patch<@code>)
      -> !fabric.patch<@code>
    fabric.dealloc %rotated : !fabric.patch<@code>
    fabric.protocol_return
  }
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
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

// CHECK: %[[REQUEST:.*]] = phys.resource_request "t_state" from @logical::@magic
// CHECK: %[[RESOURCE:.*]] = event.await %[[REQUEST]]
// CHECK: phys.resource_rotate_product %[[RESOURCE]] on
// CHECK-SAME: angle = -0.78539816339744828 : f64
// CHECK-SAME: paulis = ["Z"]
// CHECK: phys.release
