// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=bounded_root device-symbol=device graph-symbol=bounded_physical})' | FileCheck %s --check-prefix=BOUNDED
// RUN: not qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=unsupported_root device-symbol=device graph-symbol=unsupported_physical})' 2>&1 | FileCheck %s --check-prefix=REJECTED
// RUN: not qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=unknown_root device-symbol=device graph-symbol=unknown_physical})' 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=untyped_root device-symbol=device graph-symbol=untyped_physical})' 2>&1 | FileCheck %s --check-prefix=UNTYPED

module attributes {qlx.profiles = ["p2n"]} {
  lvm.domain @logical {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  fabric.code @code {
    distance = 1 : i64, partitions = {data = 1 : i64}
  }
  fabric.machine @qec {
    fabric.region @compute {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>
    }
  }

  fabric.protocol @bounded_inner : () -> () attributes {
    metadata = {runtime_replay = "bounded_p3_retry"}
  } {
    fabric.protocol_return
  }
  fabric.protocol @bounded_root : () -> () {
    fabric.call @bounded_inner() : () -> ()
    fabric.protocol_return
  }

  fabric.protocol @unsupported_inner : () -> () attributes {
    metadata = {runtime_replay = "unsupported"}
  } {
    fabric.protocol_return
  }
  fabric.protocol @unsupported_root : () -> () {
    fabric.call @unsupported_inner() : () -> ()
    fabric.protocol_return
  }

  fabric.protocol @unknown_inner : () -> () attributes {
    metadata = {runtime_replay = "future_controller"}
  } {
    fabric.protocol_return
  }
  fabric.protocol @unknown_root : () -> () {
    fabric.call @unknown_inner() : () -> ()
    fabric.protocol_return
  }

  fabric.protocol @untyped_inner : () -> () attributes {
    metadata = {runtime_replay = 1 : i64}
  } {
    fabric.protocol_return
  }
  fabric.protocol @untyped_root : () -> () {
    fabric.call @untyped_inner() : () -> ()
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

// BOUNDED: phys.graph @bounded_physical
// BOUNDED: phys.call
// BOUNDED-SAME: callee = @bounded_inner

// REJECTED: 'fabric.protocol' op native projection rejects metadata.runtime_replay = "unsupported": P3 attempt replay is not implemented
// REJECTED-NOT: phys.graph @unsupported_physical

// UNKNOWN: 'fabric.protocol' op native projection does not recognize metadata.runtime_replay = "future_controller"; expected "bounded_p3_retry"
// UNKNOWN-NOT: phys.graph @unknown_physical

// UNTYPED: 'fabric.protocol' op native projection requires metadata.runtime_replay to be a string
// UNTYPED-NOT: phys.graph @untyped_physical
