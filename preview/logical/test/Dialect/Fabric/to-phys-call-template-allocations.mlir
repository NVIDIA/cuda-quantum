// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=one device-symbol=device_one graph-symbol=one_physical})' | FileCheck %s --check-prefix=ONE
// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=two device-symbol=device_two graph-symbol=two_physical})' | FileCheck %s --check-prefix=TWO
// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=chained device-symbol=device_one graph-symbol=chained_physical})' | FileCheck %s --check-prefix=CHAINED
// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=destructive device-symbol=device_wide graph-symbol=destructive_physical})' | FileCheck %s --check-prefix=DESTRUCTIVE

module attributes {qlx.profiles = ["p2n"]} {
  lvm.domain @logical {
    lvm.space @compute {capabilities = [], capacity = 2 : i64}
  }
  fabric.code @code {
    distance = 1 : i64, partitions = {data = 1 : i64}
  }
  fabric.machine @qec {
    fabric.region @compute {
      code = @code, floorplan = #fabric.floorplan<direct, [2]>,
      role = #fabric.role<compute>
    }
  }
  fabric.protocol @uses_internal_scratch
      : (!fabric.patch<@code>) -> !fabric.patch<@code> {
  ^bb0(%block: !fabric.patch<@code>):
    %scratch = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %prepared = fabric.prep_z %scratch : !fabric.patch<@code>
    fabric.dealloc %prepared : !fabric.patch<@code>
    fabric.protocol_return %block : !fabric.patch<@code>
  }
  fabric.protocol @one : () -> () {
    %left = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %right = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %left_out = fabric.call @uses_internal_scratch(%left)
      : (!fabric.patch<@code>) -> !fabric.patch<@code>
    %right_out = fabric.call @uses_internal_scratch(%right)
      : (!fabric.patch<@code>) -> !fabric.patch<@code>
    fabric.dealloc %left_out : !fabric.patch<@code>
    fabric.dealloc %right_out : !fabric.patch<@code>
    fabric.protocol_return
  }
  fabric.protocol @two : () -> () {
    %left = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %right = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %left_out = fabric.call @uses_internal_scratch(%left)
      : (!fabric.patch<@code>) -> !fabric.patch<@code>
    %right_out = fabric.call @uses_internal_scratch(%right)
      : (!fabric.patch<@code>) -> !fabric.patch<@code>
    fabric.dealloc %left_out : !fabric.patch<@code>
    fabric.dealloc %right_out : !fabric.patch<@code>
    fabric.protocol_return
  }
  fabric.protocol @chained : () -> () {
    %block = fabric.alloc {code = @code, region = @compute}
      : !fabric.patch<@code>
    %first = fabric.call @uses_internal_scratch(%block)
      : (!fabric.patch<@code>) -> !fabric.patch<@code>
    %second = fabric.call @uses_internal_scratch(%first)
      : (!fabric.patch<@code>) -> !fabric.patch<@code>
    fabric.dealloc %second : !fabric.patch<@code>
    fabric.protocol_return
  }
  phys.machine @one_arch {
    phys.resource_class @one_qubits {
      kind = "qubit", count = 3 : i64, native_actions = []
    }
    phys.qec_binding @compute_binding {
      qec_region = @qec::@compute, resources = [@one_qubits]
    }
  }
  phys.machine @two_arch {
    phys.resource_class @two_qubits {
      kind = "qubit", count = 4 : i64, native_actions = []
    }
    phys.qec_binding @compute_binding {
      qec_region = @qec::@compute, resources = [@two_qubits]
    }
  }
  qlx.logical_to_qec @logical_to_qec {
    logical = @logical, qec = @qec,
    entries = [{logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @one_qec_to_physical {
    qec = @qec, physical = @one_arch,
    entries = [{qec = "compute", binding = "compute_binding",
                resources = ["one_qubits"]}]
  }
  qlx.qec_to_physical @two_qec_to_physical {
    qec = @qec, physical = @two_arch,
    entries = [{qec = "compute", binding = "compute_binding",
                resources = ["two_qubits"]}]
  }
  qlx.device @device_one {
    logical = @logical, qec = @qec, physical = @one_arch,
    logical_to_qec = @logical_to_qec,
    qec_to_physical = @one_qec_to_physical
  }
  qlx.device @device_two {
    logical = @logical, qec = @qec, physical = @two_arch,
    logical_to_qec = @logical_to_qec,
    qec_to_physical = @two_qec_to_physical
  }

  // Replaying a destructive call must retire one multi-carrier allocation as
  // a group, even though every carrier retains its own physical identity.
  fabric.code @wide_code {
    distance = 1 : i64, partitions = {data = 2 : i64}
  }
  fabric.machine @wide_qec {
    fabric.region @compute {
      code = @wide_code, floorplan = #fabric.floorplan<direct, [2]>,
      role = #fabric.role<compute>
    }
  }
  fabric.protocol @consume_wide
      : (!fabric.patch<@wide_code>) -> () {
  ^bb0(%block: !fabric.patch<@wide_code>):
    fabric.dealloc %block : !fabric.patch<@wide_code>
    fabric.protocol_return
  }
  fabric.protocol @destructive : () -> () {
    %left = fabric.alloc {code = @wide_code, region = @compute}
      : !fabric.patch<@wide_code>
    %right = fabric.alloc {code = @wide_code, region = @compute}
      : !fabric.patch<@wide_code>
    fabric.call @consume_wide(%left) : (!fabric.patch<@wide_code>) -> ()
    fabric.call @consume_wide(%right) : (!fabric.patch<@wide_code>) -> ()
    fabric.protocol_return
  }
  phys.machine @wide_arch {
    phys.resource_class @wide_qubits {
      kind = "qubit", count = 4 : i64, native_actions = []
    }
    phys.qec_binding @compute_binding {
      qec_region = @wide_qec::@compute, resources = [@wide_qubits]
    }
  }
  qlx.logical_to_qec @wide_logical_to_qec {
    logical = @logical, qec = @wide_qec,
    entries = [{logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @wide_qec_to_physical {
    qec = @wide_qec, physical = @wide_arch,
    entries = [{qec = "compute", binding = "compute_binding",
                resources = ["wide_qubits"]}]
  }
  qlx.device @device_wide {
    logical = @logical, qec = @wide_qec, physical = @wide_arch,
    logical_to_qec = @wide_logical_to_qec,
    qec_to_physical = @wide_qec_to_physical
  }
}

// With one scratch carrier, the second invocation has the same body-local
// resource plan and may reuse the canonical body exactly.
// ONE: phys.graph @one_physical
// ONE: "phys.call"() <{callee = @uses_internal_scratch
// ONE: phys.acquire [@one_qubits_2_alloc2]
// ONE: phys.prepare
// ONE: phys.release
// ONE: phys.call_template
// ONE-SAME: callee = @uses_internal_scratch
// ONE-NOT: "phys.call"()
// ONE: phys.release %{{.*}} : !phys.state<@one_qubits_0_alloc0>

// With another carrier available, first-fit assigns q3 to the later
// invocation.  Reusing q2's body would serialize two otherwise disjoint calls,
// so both exact bodies remain explicit.
// TWO: phys.graph @two_physical
// TWO: "phys.call"() <{callee = @uses_internal_scratch
// TWO: phys.acquire [@two_qubits_2_alloc2]
// TWO: phys.release
// TWO: "phys.call"() <{callee = @uses_internal_scratch
// TWO: phys.acquire [@two_qubits_3_alloc3]
// TWO: phys.release
// TWO-NOT: phys.call_template
// TWO: phys.release %{{.*}} : !phys.state<@two_qubits_0_alloc0>

// The second chained invocation is source-SSA-dependent on the first.  Its
// canonical q1 scratch is released before it can run, so exact reuse remains
// legal even though the otherwise-unused q2 carrier exists.
// CHAINED: phys.graph @chained_physical
// CHAINED: "phys.call"() <{callee = @uses_internal_scratch
// CHAINED: phys.acquire [@one_qubits_1_alloc1]
// CHAINED: phys.release
// CHAINED: phys.call_template
// CHAINED-SAME: callee = @uses_internal_scratch
// CHAINED-NOT: "phys.call"()
// CHAINED: phys.release %{{.*}} : !phys.state<@one_qubits_0_alloc0>

// The first call establishes the destructive template; the second invocation
// aliases it to a different two-carrier caller allocation and releases that
// allocation exactly once.
// DESTRUCTIVE: phys.graph @destructive_physical
// DESTRUCTIVE: "phys.call"({{.*}}) <{callee = @consume_wide
// DESTRUCTIVE: phys.release
// DESTRUCTIVE: phys.call_template
// DESTRUCTIVE-SAME: callee = @consume_wide
