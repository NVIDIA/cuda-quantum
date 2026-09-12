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
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  fabric.code @code {
    distance = 1 : i64,
    partitions = {data = 2 : i64}
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
    %zero = fabric.prep_z %patch : !fabric.patch<@code>
    %next = fabric.h %zero data : !fabric.patch<@code>
    fabric.dealloc %next : !fabric.patch<@code>
    fabric.protocol_return
  }
  phys.machine @arch {
    phys.action @h {
      arity = 1 : i64,
      process = "{\22kind\22:\22builtin\22,\22name\22:\22h\22,\22parameters\22:{}}"
    }
    phys.resource_class @qubits {
      kind = "qubit", count = 2 : i64, native_actions = [@h]
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

// CHECK: %[[NEXT:.*]]:2 = phys.apply @h
// CHECK-SAME: batch_lanes = 2 : i64
// CHECK-SAME: resources = [@qubits_0_alloc0, @qubits_1_alloc1]
