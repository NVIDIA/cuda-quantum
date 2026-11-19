// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=p2 device-symbol=device graph-symbol=p2_physical})' | FileCheck %s

// The non-self-dual three-bit CSS presentation prepares |+_L> from |0>^3 by
// synthesizing the LX encoder even when the empty HX family is omitted.
module attributes {qlx.profiles = ["p2n"]} {
  lvm.domain @logical {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  fabric.code @repetition {
    distance = 1 : i64,
    partitions = {data = 3 : i64, sx = 0 : i64, sz = 2 : i64},
    n = 3 : i64,
    k = 1 : i64,
    hz = [array<i64: 0, 1>, array<i64: 1, 2>],
    lx = [array<i64: 0, 1, 2>],
    lz = [array<i64: 0>]
  }
  fabric.machine @qec {
    fabric.region @compute {
      code = @repetition, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>
    }
  }
  fabric.protocol @p2 : () -> () {
    %patch = fabric.alloc {code = @repetition, region = @compute}
      : !fabric.patch<@repetition>
    %plus = fabric.prep_x %patch : !fabric.patch<@repetition>
    fabric.dealloc %plus : !fabric.patch<@repetition>
    fabric.protocol_return
  }
  phys.machine @arch {
    phys.action @h {
      arity = 1 : i64,
      process = "{\22kind\22:\22builtin\22,\22name\22:\22h\22,\22parameters\22:{}}"
    }
    phys.action @cx {
      arity = 2 : i64,
      process = "{\22kind\22:\22builtin\22,\22name\22:\22cx\22,\22parameters\22:{}}"
    }
    phys.resource_class @qubits {
      kind = "qubit", count = 5 : i64, native_actions = [@h, @cx]
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

// CHECK: phys.prepare
// CHECK-SAME: state = "zero"
// CHECK: phys.apply @h
// CHECK: phys.apply @cx
// CHECK: phys.apply @cx
// CHECK: phys.release
