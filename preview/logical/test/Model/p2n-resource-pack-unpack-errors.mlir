// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module attributes {qlx.profiles = ["p2n"]} {
  func.func @bad_anchor_successor(
      %anchor: !fabric.patch<@Steane, @steane_encoding, @epoch0>,
      %state: !fabric.resource<@t_state>) {
    // expected-error @+1 {{must preserve every anchor patch type}}
    %next, %payload = fabric.unpack_resource %state like(%anchor)
      : (!fabric.resource<@t_state>,
         !fabric.patch<@Steane, @steane_encoding, @epoch0>)
        -> (!fabric.patch<@Surface, @surface_encoding, @epoch0>,
            !fabric.patch<@BareQubit, @bare_encoding, @epoch0>)
    return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  func.func @missing_selected_block_ids(
      %anchor: !fabric.patch<@Steane, @steane_encoding, @epoch0>,
      %state: !fabric.resource<@ccz_state>) {
    // expected-error @+1 {{selected three-qubit magic-state handoff requires exact QEC block identities}}
    %next:2 = fabric.unpack_resource %state like(%anchor) {
      payload_action = #qlx.action<ccz>,
      payload_logical_blocks = array<i64: 0, 0, 0>,
      payload_logical_ports = array<i64: 0, 1, 2>
    } : (!fabric.resource<@ccz_state>,
         !fabric.patch<@Steane, @steane_encoding, @epoch0>)
      -> (!fabric.patch<@Steane, @steane_encoding, @epoch0>,
          !fabric.patch<@Steane, @steane_encoding, @epoch0>)
    return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  func.func @bad_mapping_pair(
      %anchor: !fabric.patch<@Steane, @steane_encoding, @epoch0>,
      %state: !fabric.resource<@ccz_state>) {
    // expected-error @+1 {{payload logical block and port maps must appear together}}
    %next:2 = fabric.unpack_resource %state like(%anchor) {
      payload_logical_blocks = array<i64: 0, 0, 0>
    } : (!fabric.resource<@ccz_state>,
         !fabric.patch<@Steane, @steane_encoding, @epoch0>)
      -> (!fabric.patch<@Steane, @steane_encoding, @epoch0>,
          !fabric.patch<@Steane, @steane_encoding, @epoch0>)
    return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  func.func @bad_mapping_block(
      %anchor: !fabric.patch<@Steane, @steane_encoding, @epoch0>,
      %state: !fabric.resource<@ccz_state>) {
    // expected-error @+1 {{payload logical block index is out of range}}
    %next:2 = fabric.unpack_resource %state like(%anchor) {
      payload_action = #qlx.action<ccz>,
      payload_logical_block_ids = ["anchor_block", "anchor_block", "anchor_block"],
      payload_logical_blocks = array<i64: 1, 0, 0>,
      payload_logical_ports = array<i64: 0, 1, 2>
    } : (!fabric.resource<@ccz_state>,
         !fabric.patch<@Steane, @steane_encoding, @epoch0>)
      -> (!fabric.patch<@Steane, @steane_encoding, @epoch0>,
          !fabric.patch<@Steane, @steane_encoding, @epoch0>)
    return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  fabric.encoding @steane_encoding {
    code = @Steane, profile = @steane_profile, block = "data",
    logical_ports = ["q0", "q1", "q2"]
  }
  func.func @aliased_payload_port(
      %anchor: !fabric.patch<@Steane, @steane_encoding, @epoch0>,
      %state: !fabric.resource<@ccz_state>) {
    // expected-error @+1 {{payload logical mapping must not alias a payload port}}
    %next:2 = fabric.unpack_resource %state like(%anchor) {
      payload_action = #qlx.action<ccz>,
      payload_logical_block_ids = ["anchor_block", "anchor_block", "anchor_block"],
      payload_logical_blocks = array<i64: 0, 0, 0>,
      payload_logical_ports = array<i64: 0, 0, 1>
    } : (!fabric.resource<@ccz_state>,
         !fabric.patch<@Steane, @steane_encoding, @epoch0>)
      -> (!fabric.patch<@Steane, @steane_encoding, @epoch0>,
          !fabric.patch<@Steane, @steane_encoding, @epoch0>)
    return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  func.func @legacy_injection(
      %patch: !fabric.patch<@Steane, @steane_encoding, @epoch0>,
      %state: !fabric.resource<@t_state>) {
    // expected-error @+1 {{is legacy logical intent and is not legal in canonical P2/P3}}
    %next = fabric.inject %patch, %state {
      protocol = #fabric.spec_only<"legacy-t-injection">,
      gate = "t"
    } : (!fabric.patch<@Steane, @steane_encoding, @epoch0>,
         !fabric.resource<@t_state>)
      -> !fabric.patch<@Steane, @steane_encoding, @epoch0>
    return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  func.func @bad_kind(
      %payload: !fabric.patch<@Steane, @steane_encoding, @epoch0>) {
    // expected-error @+1 {{resource_kind must match the packed result resource type}}
    %state = fabric.pack_resource (%payload) as @t_state
      {payload_encodings = [@steane_encoding]}
      : (!fabric.patch<@Steane, @steane_encoding, @epoch0>)
        -> !fabric.resource<@ccz_state>
    return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  func.func @missing_multi_payload_maps(
      %a: !fabric.patch<@Toric, @toric_encoding, @epoch0>,
      %b: !fabric.patch<@Toric, @toric_encoding, @epoch0>,
      %state: !fabric.resource<@ccz_state>) {
    // expected-error @+1 {{multi-payload unpack requires an action and exact logical maps}}
    %next:4 = fabric.unpack_resource %state like(%a, %b)
      : (!fabric.resource<@ccz_state>,
         !fabric.patch<@Toric, @toric_encoding, @epoch0>,
         !fabric.patch<@Toric, @toric_encoding, @epoch0>)
      -> (!fabric.patch<@Toric, @toric_encoding, @epoch0>,
          !fabric.patch<@Toric, @toric_encoding, @epoch0>,
          !fabric.patch<@Toric, @toric_encoding, @epoch0>,
          !fabric.patch<@Toric, @toric_encoding, @epoch0>)
    return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  func.func @wrong_action_arity(
      %a: !fabric.patch<@Toric, @toric_encoding, @epoch0>,
      %state: !fabric.resource<@ccz_state>) {
    // expected-error @+1 {{payload logical maps must exactly cover the action arity}}
    %next:2 = fabric.unpack_resource %state like(%a) {
      payload_action = #qlx.action<ccz>,
      payload_logical_block_ids = ["a_block", "a_block"],
      payload_logical_blocks = array<i64: 0, 0>,
      payload_logical_ports = array<i64: 0, 1>
    } : (!fabric.resource<@ccz_state>,
         !fabric.patch<@Toric, @toric_encoding, @epoch0>)
      -> (!fabric.patch<@Toric, @toric_encoding, @epoch0>,
          !fabric.patch<@Toric, @toric_encoding, @epoch0>)
    return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  fabric.encoding @toric_encoding {
    code = @Toric, profile = @toric_profile, block = "data",
    logical_ports = ["q0", "q1", "q2"]
  }
  func.func @missing_payload_block(
      %a: !fabric.patch<@Toric, @toric_encoding, @epoch0>,
      %b: !fabric.patch<@Toric, @toric_encoding, @epoch0>,
      %state: !fabric.resource<@ccz_state>) {
    // expected-error @+1 {{payload logical maps must cover every payload block}}
    %next:4 = fabric.unpack_resource %state like(%a, %b) {
      payload_action = #qlx.action<ccz>,
      payload_logical_block_ids = ["a_block", "a_block", "a_block"],
      payload_logical_blocks = array<i64: 0, 0, 0>,
      payload_logical_ports = array<i64: 0, 1, 2>
    } : (!fabric.resource<@ccz_state>,
         !fabric.patch<@Toric, @toric_encoding, @epoch0>,
         !fabric.patch<@Toric, @toric_encoding, @epoch0>)
      -> (!fabric.patch<@Toric, @toric_encoding, @epoch0>,
          !fabric.patch<@Toric, @toric_encoding, @epoch0>,
          !fabric.patch<@Toric, @toric_encoding, @epoch0>,
          !fabric.patch<@Toric, @toric_encoding, @epoch0>)
    return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  fabric.protocol @mismatched_action :
      (!fabric.patch<@Toric, @toric_encoding, @epoch0>,
       !fabric.resource<@ccz_state>)
      -> !fabric.patch<@Toric, @toric_encoding, @epoch0>
      attributes {objective = #qlx.action<cz>} {
  ^bb0(%a: !fabric.patch<@Toric, @toric_encoding, @epoch0>,
       %state: !fabric.resource<@ccz_state>):
    // expected-error @+1 {{payload_action must match the enclosing protocol objective}}
    %next:2 = fabric.unpack_resource %state like(%a) {
      payload_action = #qlx.action<ccz>,
      payload_logical_block_ids = ["a_block", "a_block", "a_block"],
      payload_logical_blocks = array<i64: 0, 0, 0>,
      payload_logical_ports = array<i64: 0, 1, 2>
    } : (!fabric.resource<@ccz_state>,
         !fabric.patch<@Toric, @toric_encoding, @epoch0>)
      -> (!fabric.patch<@Toric, @toric_encoding, @epoch0>,
          !fabric.patch<@Toric, @toric_encoding, @epoch0>)
    fabric.protocol_return %next#0
      : !fabric.patch<@Toric, @toric_encoding, @epoch0>
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  fabric.encoding @toric_encoding {
    code = @Toric, profile = @toric_profile, block = "data",
    logical_ports = ["q0", "q1"]
  }
  func.func @port_exceeds_encoding(
      %a: !fabric.patch<@Toric, @toric_encoding, @epoch0>,
      %state: !fabric.resource<@ccz_state>) {
    // expected-error @+1 {{payload logical port index exceeds the encoding logical capacity}}
    %next:2 = fabric.unpack_resource %state like(%a) {
      payload_action = #qlx.action<ccz>,
      payload_logical_block_ids = ["a_block", "a_block", "a_block"],
      payload_logical_blocks = array<i64: 0, 0, 0>,
      payload_logical_ports = array<i64: 0, 1, 2>
    } : (!fabric.resource<@ccz_state>,
         !fabric.patch<@Toric, @toric_encoding, @epoch0>)
      -> (!fabric.patch<@Toric, @toric_encoding, @epoch0>,
          !fabric.patch<@Toric, @toric_encoding, @epoch0>)
    return
  }
}
