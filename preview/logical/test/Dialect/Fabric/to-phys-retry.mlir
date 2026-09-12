// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=p2 device-symbol=device})' | FileCheck %s
// RUN: qlx-opt %s --fabric-count='root=p2 device=device result=counts' | FileCheck %s --check-prefix=COUNT

module attributes {qlx.profiles = ["p2n"]} {
  phys.action @x {
    arity = 1 : i64,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22x\22,\22parameters\22:{}}"
  }
  phys.instrument @mpp {
    kind = "measure_product", variadic, record_schema = "bit",
    preserves_inputs,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22measure_product\22,\22parameters\22:{}}"
  }
  lvm.domain @logical {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  fabric.code @code {
    distance = 1 : i64, k = 1 : i64, n = 1 : i64, r = 0 : i64,
    partitions = {data = 1 : i64},
    lx = [array<i64: 0>], lz = [array<i64: 0>]
  }
  fabric.gadget_spec @attempt_spec for @objective
      : (!fabric.patch<@code>) -> (!fabric.patch<@code>, i1) {
    encodings = [],
    outcome_map = {
      records = ["ready.data0"], rows = dense<1> : tensor<1x1xi1>,
      constants = array<i64: 0>, input_syndromes = [[]],
      roles = [["success"]]
    },
    record_schema = ["ready.data0"]
  }
  fabric.gadget @attempt(%patch: !fabric.patch<@code>)
      -> (!fabric.patch<@code>, i1) {
    %next, %bits = fabric.mz %patch data [0] {record = "ready"}
      : !fabric.patch<@code> -> tensor<1xi1>
    %mismatch = fabric.parity %bits : (tensor<1xi1>) -> i1
    %selected = cflow.if %mismatch -> !fabric.patch<@code> {
      %updated = fabric.x %next data : !fabric.patch<@code>
      cflow.yield %updated : !fabric.patch<@code>
    } else {
      cflow.yield %next : !fabric.patch<@code>
    }
    fabric.return %selected, %mismatch : !fabric.patch<@code>, i1
  } {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @attempt_profile for @attempt {
    fabric.success {records = ["attempt.ready.data0"]}
  }
  fabric.gadget_profile @attempt_profile_again for @attempt
      attributes {metadata = {variant = "second"}} {
    fabric.success {records = ["attempt.ready.data0"]}
  }
  fabric.gadget_spec @direct_attempt_spec for @objective
      : (!fabric.patch<@code>) -> (!fabric.patch<@code>, i1) {
    encodings = [],
    outcome_map = {
      records = ["ready.outcome"], rows = dense<1> : tensor<1x1xi1>,
      constants = array<i64: 0>
    },
    record_schema = ["ready.outcome"]
  }
  fabric.gadget @direct_attempt(%patch: !fabric.patch<@code>)
      -> (!fabric.patch<@code>, i1) {
    %next, %ready = fabric.measure_product %patch {
      logical_indices = array<i64: 0>, patch_indices = array<i64: 0>,
      pauli_product = "Z", record = "ready"
    } : (!fabric.patch<@code>) -> (!fabric.patch<@code>, i1)
    fabric.return %next, %ready : !fabric.patch<@code>, i1
  } {realization_boundary = {}, spec = @direct_attempt_spec}
  fabric.gadget_profile @direct_attempt_profile for @direct_attempt {
    fabric.success {constant = true,
                    records = ["direct_attempt.ready.outcome"]}
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
    %attempted, %ok = fabric.call @attempt(%patch) {
      profile = @attempt_profile
    } : (!fabric.patch<@code>) -> (!fabric.patch<@code>, i1)
    %decision = fabric.all_false %ok : (i1) -> i1
    %retried = fabric.retry %decision carries (%attempted) {
      attempt = @attempt, exhaustion = "abort", max_attempts = 3 : i64,
      profile = @attempt_profile
    } : (!fabric.patch<@code>) -> !fabric.patch<@code>
    %attempted_again, %ok_again = fabric.call @attempt(%retried) {
      profile = @attempt_profile_again
    } : (!fabric.patch<@code>) -> (!fabric.patch<@code>, i1)
    %decision_again = fabric.all_false %ok_again : (i1) -> i1
    %retried_again = fabric.retry %decision_again carries (%attempted_again) {
      attempt = @attempt, exhaustion = "abort", max_attempts = 3 : i64,
      profile = @attempt_profile_again
    } : (!fabric.patch<@code>) -> !fabric.patch<@code>
    %direct_attempted, %direct_ok = fabric.call @direct_attempt(%retried_again) {
      profile = @direct_attempt_profile
    } : (!fabric.patch<@code>) -> (!fabric.patch<@code>, i1)
    %direct_retried = fabric.retry %direct_ok carries (%direct_attempted) {
      attempt = @direct_attempt, exhaustion = "abort", max_attempts = 3 : i64,
      profile = @direct_attempt_profile
    } : (!fabric.patch<@code>) -> !fabric.patch<@code>
    fabric.dealloc %direct_retried : !fabric.patch<@code>
    fabric.protocol_return
  }
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = [@x],
      native_instruments = [@mpp]
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

// CHECK: %[[CALL:.*]]:2 = "phys.call"
// CHECK-SAME: callee = @attempt
// CHECK-SAME: profile = @attempt_profile
// CHECK: phys.measure @measure_z_instrument
// CHECK: phys.condition
// CHECK: cflow.if
// CHECK-COUNT-1: phys.apply @x
// CHECK: cflow.yield
// CHECK: cflow.yield
// CHECK: phys.yield
// CHECK: %[[DECISION:.*]] = phys.all_false %[[CALL]]#1
// CHECK: phys.barrier {{.*}}domains = ["clock", "factory"]
// CHECK: %[[RETRIED:.*]] = phys.retry %[[DECISION]] carries(%[[CALL]]#0)
// CHECK-SAME: attempt = @attempt
// CHECK-SAME: exhaustion = "abort"
// CHECK-SAME: max_attempts = 3 : i64
// CHECK-SAME: profile = @attempt_profile
// CHECK: %[[TEMPLATE:.*]]:2 = phys.call_template %[[RETRIED]]
// CHECK-SAME: callee = @attempt
// CHECK-SAME: profile = @attempt_profile_again

// COUNT: retry_demands = [
// COUNT-SAME: {attempt = @attempt, attempt_operation_sites = 3 : i64,
// COUNT-SAME: profile = @attempt_profile,
// COUNT-SAME: {attempt = @attempt, attempt_operation_sites = 3 : i64,
// COUNT-SAME: profile = @attempt_profile_again,
// CHECK-SAME: template_event = "call1"
// CHECK: %[[DECISION_AGAIN:.*]] = phys.all_false %[[TEMPLATE]]#1
// CHECK: phys.barrier {{.*}}domains = ["clock", "factory"]
// CHECK: phys.retry %[[DECISION_AGAIN]] carries(%[[TEMPLATE]]#0)
// CHECK-SAME: attempt = @attempt
// CHECK-SAME: exhaustion = "abort"
// CHECK-SAME: max_attempts = 3 : i64
// CHECK-SAME: profile = @attempt_profile_again
// CHECK: %[[DIRECT_CALL:.*]]:2 = "phys.call"
// CHECK-SAME: callee = @direct_attempt
// CHECK: %[[DIRECT_DECISION:.*]] = phys.condition %[[DIRECT_CALL]]#1
// CHECK: phys.retry %[[DIRECT_DECISION]] carries(%[[DIRECT_CALL]]#0)
// CHECK-SAME: attempt = @direct_attempt
// CHECK-SAME: profile = @direct_attempt_profile
// CHECK: phys.barrier {{.*}}domains = ["clock", "factory"]
// CHECK: phys.release
// CHECK: phys.selection_sidecar @p2_physical_selection0 for @p2_physical
// CHECK-SAME: projection_indices = array<i64: 0>
// CHECK-SAME: source_instance = "attempt.call0"
// CHECK-SAME: source_profile = @attempt_profile
// CHECK-SAME: source_records = ["attempt.ready.data0"]
// CHECK: phys.record_projection @p2_physical_record_projection for @p2_physical from @p2
// CHECK-SAME: instance = "attempt.call0"
// CHECK-SAME: source_record = "attempt.ready.data0"
