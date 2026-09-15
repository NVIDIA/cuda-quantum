// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=p2 device-symbol=device graph-symbol=p2_physical})' | FileCheck %s

// The two reachable actions belong to disjoint carrier homes.  Native
// preflight must validate h only on @h_qubits and x only on @x_qubits; taking
// the global resource-class x action Cartesian product rejects this valid
// heterogeneous machine.
module attributes {qlx.profiles = ["p2n"]} {
  lvm.domain @logical {
    lvm.space @left {capabilities = [], capacity = 1 : i64}
    lvm.space @right {capabilities = [], capacity = 1 : i64}
  }
  fabric.code @code {
    distance = 1 : i64,
    partitions = {data = 1 : i64}
  }
  fabric.machine @qec {
    fabric.region @left {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>
    }
    fabric.region @right {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>
    }
  }
  fabric.protocol @p2 : () -> () {
    %left = fabric.alloc {code = @code, region = @left}
      : !fabric.patch<@code>
    %left_next = fabric.h %left data : !fabric.patch<@code>
    fabric.dealloc %left_next : !fabric.patch<@code>
    %right = fabric.alloc {code = @code, region = @right}
      : !fabric.patch<@code>
    %right_next = fabric.x %right data : !fabric.patch<@code>
    fabric.dealloc %right_next : !fabric.patch<@code>
    fabric.protocol_return
  }
  phys.action @h {
    arity = 1 : i64,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22h\22,\22parameters\22:{}}"
  }
  phys.action @x {
    arity = 1 : i64,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22x\22,\22parameters\22:{}}"
  }
  phys.machine @arch {
    phys.resource_class @h_qubits {
      kind = "qubit", count = 1 : i64, native_actions = [@h]
    }
    phys.resource_class @x_qubits {
      kind = "qubit", count = 1 : i64, native_actions = [@x]
    }
    phys.qec_binding @left_binding {
      qec_region = @qec::@left, resources = [@h_qubits]
    }
    phys.qec_binding @right_binding {
      qec_region = @qec::@right, resources = [@x_qubits]
    }
  }
  qlx.logical_to_qec @logical_to_qec {
    logical = @logical, qec = @qec,
    entries = [{logical = "left", qec = "left"},
               {logical = "right", qec = "right"}]
  }
  qlx.qec_to_physical @qec_to_physical {
    qec = @qec, physical = @arch,
    entries = [{qec = "left", binding = "left_binding",
                resources = ["h_qubits"]},
               {qec = "right", binding = "right_binding",
                resources = ["x_qubits"]}]
  }
  qlx.device @device {
    logical = @logical, qec = @qec, physical = @arch,
    logical_to_qec = @logical_to_qec,
    qec_to_physical = @qec_to_physical
  }
}

// CHECK: phys.apply @h
// CHECK-SAME: resources = [@h_qubits_0_alloc0]
// CHECK: phys.apply @x
// CHECK-SAME: resources = [@x_qubits_0_alloc1]
