// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: not qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=p2 device-symbol=device graph-symbol=p2_physical preflight-only=true})' 2>&1 | FileCheck %s

module attributes {qlx.profiles = ["p2n"]} {
  lvm.domain @logical {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
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
    %patch = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    fabric.dealloc %patch : !fabric.patch<@code>
    fabric.protocol_return
  }
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
    phys.topology @carrier_topology {
      kind = "adjacency", num_nodes = 1 : i64, edges = [], strict
    }
    phys.patch_topology @patches {
      capacity = 1 : i64,
      carrier_groups = [array<i64: 0>],
      categories = ["data"], edges = [],
      carrier_topology = @carrier_topology,
      resource_class = @qubits
    }
    phys.qec_binding @compute_binding {
      qec_region = @qec::@compute, resources = [@qubits],
      topology = @carrier_topology, patch_topology = @patches
    }
  }
  qlx.logical_to_qec @logical_to_qec {
    logical = @logical, qec = @qec,
    entries = [{logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @qec_to_physical {
    qec = @qec, physical = @arch,
    entries = [{qec = "compute", binding = "compute_binding",
                resources = ["qubits"], topology = "carrier_topology",
                patch_topology = "patches"}]
  }
  qlx.device @device {
    logical = @logical, qec = @qec, physical = @arch,
    logical_to_qec = @logical_to_qec,
    qec_to_physical = @qec_to_physical
  }
}

// CHECK: 'phys.qec_binding' op native projection does not yet implement selected patch-topology mapping
// CHECK-NOT: phys.graph @p2_physical
