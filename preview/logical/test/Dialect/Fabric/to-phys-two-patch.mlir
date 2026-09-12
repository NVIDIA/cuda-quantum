// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=p2 device-symbol=device})' | FileCheck %s

module attributes {qlx.profiles = ["p2n"]} {
  phys.action @cx {
    arity = 2 : i64,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22cx\22,\22parameters\22:{}}"
  }
  phys.action @cz {
    arity = 2 : i64,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22cz\22,\22parameters\22:{}}"
  }
  lvm.domain @logical {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  fabric.code @code {
    distance = 1 : i64, k = 1 : i64, n = 2 : i64, r = 0 : i64,
    partitions = {data = 2 : i64, sx = 1 : i64},
    lx = [array<i64: 0>], lz = [array<i64: 0>]
  }
  fabric.code_profile @profile {
    code = @code, distance_claim = 1 : i64, distance_status = "claimed"
  }
  fabric.encoding @encoding {
    block = "block0", code = @code, logical_ports = ["q0"], profile = @profile
  }
  fabric.patch_transform @frame from @encoding to @encoding {
    destination_roles = {
      active = array<i64: 0, 1>, dormant = array<i64>, measured = array<i64>,
      reset = array<i64>, scratch = array<i64>
    },
    destination_support = array<i64: 0, 1>,
    evidence = "exact_identity_frame",
    frame_partitions = {data = 2 : i64},
    logical_map = array<i64: 0>,
    source_roles = {
      active = array<i64: 0, 1>, dormant = array<i64>, measured = array<i64>,
      reset = array<i64>, scratch = array<i64>
    },
    source_support = array<i64: 0, 1>
  }
  fabric.machine @qec {
    fabric.region @compute {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>
    }
  }
  fabric.protocol @p2 : () -> () {
    %patch = fabric.alloc {code = @code, encoding = @encoding, region = @compute}
      : !fabric.patch<@code, @encoding>
    %rotated = fabric.rotate_product %patch {
      angle = 5.000000e-01 : f64,
      logical_indices = array<i64: 0>,
      patch_indices = array<i64: 0>,
      pauli_product = "Y", synthesis = "native"
    } : (!fabric.patch<@code, @encoding>) -> !fabric.patch<@code, @encoding>
    %frame = fabric.transform_begin %rotated using @frame
      : (!fabric.patch<@code, @encoding>) -> !fabric.patch_frame<@frame>
    %cx = fabric.cx %frame data -> data {pairs = "0:1"}
      : (!fabric.patch_frame<@frame>) -> !fabric.patch_frame<@frame>
    %cz = fabric.cz %cx data -> data {pairs = "1:0"}
      : (!fabric.patch_frame<@frame>) -> !fabric.patch_frame<@frame>
    %measured, %bits = fabric.mz %cz data [0, 1] {record = "readout"}
      : !fabric.patch_frame<@frame> -> tensor<2xi1>
    %parity = fabric.parity %bits : (tensor<2xi1>) -> i1
    %all_zero = fabric.all_zero %bits : tensor<2xi1> -> i1
    %done = fabric.transform_end %measured using @frame
      : (!fabric.patch_frame<@frame>) -> !fabric.patch<@code, @encoding>
    fabric.dealloc %done : !fabric.patch<@code, @encoding>
    fabric.protocol_return
  }
  phys.machine @arch {
    phys.resource_class @qubits {
      capabilities = ["qlx.physical/native_pauli_product_rotation"],
      kind = "qubit", count = 3 : i64, native_actions = [@cx, @cz]
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

// CHECK: phys.graph @p2_physical
// CHECK: phys.rotate_product
// CHECK-SAME: angle = 5.000000e-01 : f64
// CHECK-SAME: paulis = ["Y"]
// CHECK: %[[CX:.*]]:2 = phys.apply @cx
// CHECK-SAME: resources = [@qubits_0_alloc0, @qubits_1_alloc1]
// CHECK: %[[CZ:.*]]:2 = phys.apply @cz
// CHECK-SAME: resources = [@qubits_1_alloc1, @qubits_0_alloc0]
// CHECK: phys.measure @measure_z_instrument
// CHECK-SAME: record_id = "p2_physical.readout.0"
// CHECK: phys.measure @measure_z_instrument
// CHECK-SAME: record_id = "p2_physical.readout.1"
// CHECK: phys.condition
// CHECK: phys.condition
// CHECK: phys.xor
// CHECK: phys.condition
// CHECK: phys.condition
// CHECK: phys.all_false
// CHECK: phys.release %{{.*}}, %{{.*}}, %{{.*}}
