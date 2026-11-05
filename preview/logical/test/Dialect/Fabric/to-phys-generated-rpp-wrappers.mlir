// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=p2 device-symbol=device graph-symbol=p2_physical})' | FileCheck %s --implicit-check-not='phys.call_template {callee = @generated_rpp_adapter' --implicit-check-not='phys.call_template {callee = @generated_empty_impl'

// Generated RPP adapters are provenance boundaries rather than reusable call
// templates, even when the optional action_site witness is absent. The
// fixed-body implementation behind such an adapter may still be shared.
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
    %left_out = fabric.call @generated_rpp_adapter(%left_zero)
      : (!fabric.patch<@code>) -> !fabric.patch<@code>
    %right_out = fabric.call @generated_rpp_adapter(%right_zero)
      : (!fabric.patch<@code>) -> !fabric.patch<@code>
    fabric.dealloc %left_out : !fabric.patch<@code>
    fabric.dealloc %right_out : !fabric.patch<@code>
    fabric.call @generated_empty_adapter() : () -> ()
    fabric.call @generated_empty_adapter() : () -> ()
    fabric.protocol_return
  }
  fabric.gadget @generated_rpp_adapter(%patch: !fabric.patch<@code>)
      -> !fabric.patch<@code> {
    %prepared = fabric.call @generated_rpp_impl(%patch)
      : (!fabric.patch<@code>) -> !fabric.patch<@code>
    fabric.return %prepared : !fabric.patch<@code>
  } {
    generated_by = @rpp_compiler,
    specialization = {
      x_mask = 1 : i64,
      z_mask = 0 : i64,
      sign = 1 : i64,
      angle = 3.000000e-01 : f64,
      effective_angle = 3.000000e-01 : f64,
      precision = 1.000000e-04 : f64,
      rpp_strategy = "synthesis",
      angle_convention = "exp(-i*theta*P/2)"
    }
  }
  fabric.gadget @generated_rpp_impl(%patch: !fabric.patch<@code>)
      -> !fabric.patch<@code> {
    %prepared = fabric.prep_x %patch : !fabric.patch<@code>
    fabric.return %prepared : !fabric.patch<@code>
  } {
    // The adapter's typed specialization, never this descriptive string,
    // controls whether the invocation-specific outer boundary may be shared.
    metadata = {compiler = "renamed.compiler.with.no.semantic_weight"}
  }
  fabric.gadget @generated_empty_adapter() {
    fabric.call @generated_empty_impl() : () -> ()
    fabric.return
  } {
    generated_by = @rpp_compiler,
    specialization = {
      x_mask = 1 : i64,
      z_mask = 0 : i64,
      sign = 1 : i64,
      angle = 3.000000e-01 : f64,
      effective_angle = 3.000000e-01 : f64,
      precision = 1.000000e-04 : f64,
      rpp_strategy = "synthesis",
      angle_convention = "exp(-i*theta*P/2)"
    }
  }
  fabric.gadget @generated_empty_impl() {
    fabric.return
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

// CHECK: callee = @generated_rpp_adapter
// CHECK: callee = @generated_rpp_impl
// CHECK: callee = @generated_rpp_adapter
// CHECK: phys.call_template {callee = @generated_rpp_impl
// CHECK: callee = @generated_empty_adapter
// CHECK: callee = @generated_empty_impl
// CHECK: callee = @generated_empty_adapter
// CHECK: callee = @generated_empty_impl
