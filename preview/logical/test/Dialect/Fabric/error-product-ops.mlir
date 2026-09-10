// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file -verify-diagnostics %s

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.gadget @bad_lengths(%p: !fabric.patch<@c>) -> (!fabric.patch<@c>, i1) {
  // expected-error@+1 {{patch_indices, logical_indices, and pauli_product must have equal length}}
  %p1, %m = fabric.measure_product %p {
    logical_indices = array<i64: 0>,
    patch_indices = array<i64: 0, 0>,
    pauli_product = "Z"
  } : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
  fabric.return %p1, %m : !fabric.patch<@c>, i1
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.gadget @bad_synthesis(%p: !fabric.patch<@c>) -> !fabric.patch<@c> {
  // expected-error@+1 {{synthesis must be one of auto, native, decompose}}
  %p1 = fabric.rotate_product %p {
    angle = 1.000000e+00 : f64,
    logical_indices = array<i64: 0>,
    patch_indices = array<i64: 0>,
    pauli_product = "Z",
    synthesis = "maybe"
  } : (!fabric.patch<@c>) -> !fabric.patch<@c>
  fabric.return %p1 : !fabric.patch<@c>
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.gadget @bare_sign(%p: !fabric.patch<@c>) -> (!fabric.patch<@c>, i1) {
  // expected-error@+1 {{pauli_product must contain at least one Pauli term after the optional leading '-' sign}}
  %p1, %m = fabric.measure_product %p {
    logical_indices = array<i64>,
    patch_indices = array<i64>,
    pauli_product = "-"
  } : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
  fabric.return %p1, %m : !fabric.patch<@c>, i1
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 2 : i64},
  k = 2 : i64
}

fabric.gadget @interior_sign(%p: !fabric.patch<@c>) -> (!fabric.patch<@c>, i1) {
  // expected-error@+1 {{pauli_product must contain only X/Y/Z after the optional leading '-'; got '-'}}
  %p1, %m = fabric.measure_product %p {
    logical_indices = array<i64: 0, 1>,
    patch_indices = array<i64: 0, 0>,
    pauli_product = "-Z-"
  } : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
  fabric.return %p1, %m : !fabric.patch<@c>, i1
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 2 : i64}
}

fabric.gadget @mpp_bare_sign(%p: !fabric.patch<@c>) -> (!fabric.patch<@c>, tensor<1xi1>) {
  // expected-error@+1 {{paulis must contain at least one Pauli after the optional leading '-' sign}}
  %p1, %bits = fabric.mpp %p data indices [0] paulis "-" {record = "signed"}
      : !fabric.patch<@c> -> tensor<1xi1>
  fabric.return %p1, %bits : !fabric.patch<@c>, tensor<1xi1>
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 1 : i64},
  k = 1 : i64
}

fabric.protocol @bad_resource_logical_port :
    (!fabric.patch<@c>, !fabric.resource<@t_state>) -> !fabric.patch<@c> {
^bb0(%p: !fabric.patch<@c>, %raw: !fabric.resource<@t_state>):
  // expected-error@+1 {{logical_indices[0] = 999 out of range for code @c protected/gauge ports [0, 1)}}
  %next = fabric.resource_rotate_product %raw on %p {
    angle = 7.8539816339744828e-01 : f64,
    logical_indices = array<i64: 999>,
    patch_indices = array<i64: 0>,
    pauli_product = "Z",
    subsystem_indices = array<i64: 999>,
    subsystem_kinds = ["protected"]
  } : (!fabric.resource<@t_state>, !fabric.patch<@c>) -> !fabric.patch<@c>
  fabric.protocol_return %next : !fabric.patch<@c>
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 2 : i64},
  k = 2 : i64
}

fabric.gadget @inconsistent_subsystem(%p: !fabric.patch<@c>)
    -> (!fabric.patch<@c>, i1) {
  // expected-error@+1 {{logical_indices[0] = 1 is inconsistent with protected subsystem index 0; expected 0}}
  %next, %outcome = fabric.measure_product %p {
    logical_indices = array<i64: 1>,
    patch_indices = array<i64: 0>,
    pauli_product = "Z",
    subsystem_indices = array<i64: 0>,
    subsystem_kinds = ["protected"]
  } : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
  fabric.return %next, %outcome : !fabric.patch<@c>, i1
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 1 : i64},
  k = 1 : i64
}

fabric.gadget @duplicate_product_address(%p: !fabric.patch<@c>)
    -> (!fabric.patch<@c>, i1) {
  // expected-error@+1 {{product repeats logical address (patch 0, logical 0)}}
  %next, %outcome = fabric.measure_product %p {
    logical_indices = array<i64: 0, 0>,
    patch_indices = array<i64: 0, 0>,
    pauli_product = "ZZ",
    subsystem_indices = array<i64: 0, 0>,
    subsystem_kinds = ["protected", "protected"]
  } : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
  fabric.return %next, %outcome : !fabric.patch<@c>, i1
}
