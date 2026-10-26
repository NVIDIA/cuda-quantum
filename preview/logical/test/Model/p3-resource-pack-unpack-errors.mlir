// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module attributes {qlx.profiles = ["p3"]} {
  func.func @bad_carrier_type(
      %state: !phys.resource_payload<@t_state>,
      %carrier: !phys.state<@magic0>) {
    // expected-error @+1 {{must preserve destination carrier state types}}
    %payload = phys.unpack_resource %state into(%carrier)
      {encoding = @steane_encoding}
      : (!phys.resource_payload<@t_state>, !phys.state<@magic0>)
        -> !phys.state<@magic1>
    return
  }
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  fabric.code @Steane {
    distance = 1 : i64, n = 1 : i64, partitions = {data = 1 : i64}
  }
  fabric.encoding @steane_encoding {
    code = @Steane, profile = @steane_profile, block = "data",
    logical_ports = ["q0", "q1", "q2"]
  }
  func.func @missing_selected_block_ids(
      %state: !phys.resource_payload<@ccz_state>,
      %carrier: !phys.state<@carrier>) {
    // expected-error @+1 {{selected CCZ payload handoff requires exact QEC block identities}}
    %next = phys.unpack_resource %state into(%carrier) {
      encoding = @steane_encoding,
      payload_action = #qlx.action<ccz>,
      payload_carrier_segments = array<i64: 0, 1>,
      payload_logical_blocks = array<i64: 0, 0, 0>,
      payload_logical_ports = array<i64: 0, 1, 2>
    } : (!phys.resource_payload<@ccz_state>, !phys.state<@carrier>)
      -> !phys.state<@carrier>
    return
  }
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  func.func @bad_segments(
      %state: !phys.resource_payload<@ccz_state>,
      %left: !phys.state<@left0>,
      %right: !phys.state<@right0>) {
    // expected-error @+1 {{payload carrier segments must start at zero and cover all carriers}}
    %next:2 = phys.unpack_resource %state into(%left, %right) {
      encoding = @toric_encoding,
      payload_carrier_segments = array<i64: 1, 2>
    } : (!phys.resource_payload<@ccz_state>,
         !phys.state<@left0>, !phys.state<@right0>)
      -> (!phys.state<@left0>, !phys.state<@right0>)
    return
  }
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  fabric.code @Toric {
    distance = 1 : i64, n = 1 : i64, partitions = {data = 1 : i64}
  }
  fabric.encoding @toric_encoding {
    code = @Toric, profile = @toric_profile, block = "data",
    logical_ports = ["q0", "q1", "q2"]
  }
  func.func @bad_logical_block(
      %state: !phys.resource_payload<@ccz_state>,
      %carrier: !phys.state<@left0>) {
    // expected-error @+1 {{payload logical block index is out of range}}
    %next = phys.unpack_resource %state into(%carrier) {
      encoding = @toric_encoding,
      payload_carrier_segments = array<i64: 0, 1>,
      payload_action = #qlx.action<ccz>,
      payload_logical_block_ids = ["carrier_block", "carrier_block", "carrier_block"],
      payload_logical_blocks = array<i64: 1, 0, 0>,
      payload_logical_ports = array<i64: 0, 0, 1>
    } : (!phys.resource_payload<@ccz_state>, !phys.state<@left0>)
      -> !phys.state<@left0>
    return
  }
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  func.func @legacy_consumption(
      %state: !phys.resource_payload<@t_state>,
      %carrier: !phys.state<@data0>) {
    // expected-error @+1 {{is legacy logical intent and is not legal in canonical P3}}
    %next = phys.consume_resource %state with @t(%carrier)
      : (!phys.resource_payload<@t_state>, !phys.state<@data0>)
        -> !phys.state<@data0>
    return
  }
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  func.func @bad_kind(%carrier: !phys.state<@magic0>) {
    // expected-error @+1 {{resource_kind must match the result resource type}}
    %state = phys.pack_resource(%carrier) as @t_state
      {encoding = @steane_encoding}
      : (!phys.state<@magic0>) -> !phys.resource_payload<@ccz_state>
    return
  }
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  func.func @maps_without_segments(
      %state: !phys.resource_payload<@ccz_state>,
      %carrier: !phys.state<@left0>) {
    // expected-error @+1 {{payload logical maps require payload carrier segments}}
    %next = phys.unpack_resource %state into(%carrier) {
      encoding = @toric_encoding,
      payload_action = #qlx.action<ccz>,
      payload_logical_block_ids = ["carrier_block", "carrier_block", "carrier_block"],
      payload_logical_blocks = array<i64: 0, 0, 0>,
      payload_logical_ports = array<i64: 0, 1, 2>
    } : (!phys.resource_payload<@ccz_state>, !phys.state<@left0>)
      -> !phys.state<@left0>
    return
  }
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  fabric.code @Toric {
    distance = 1 : i64, n = 1 : i64, partitions = {data = 1 : i64}
  }
  fabric.encoding @toric_encoding {
    code = @Toric, profile = @toric_profile, block = "data",
    logical_ports = ["q0", "q1", "q2"]
  }
  func.func @wrong_action_arity(
      %state: !phys.resource_payload<@ccz_state>,
      %carrier: !phys.state<@left0>) {
    // expected-error @+1 {{payload logical maps must exactly cover the action arity}}
    %next = phys.unpack_resource %state into(%carrier) {
      encoding = @toric_encoding,
      payload_action = #qlx.action<ccz>,
      payload_carrier_segments = array<i64: 0, 1>,
      payload_logical_block_ids = ["carrier_block", "carrier_block"],
      payload_logical_blocks = array<i64: 0, 0>,
      payload_logical_ports = array<i64: 0, 1>
    } : (!phys.resource_payload<@ccz_state>, !phys.state<@left0>)
      -> !phys.state<@left0>
    return
  }
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  fabric.code @Toric {
    distance = 1 : i64, n = 1 : i64, partitions = {data = 1 : i64}
  }
  fabric.encoding @toric_encoding {
    code = @Toric, profile = @toric_profile, block = "data",
    logical_ports = ["q0", "q1", "q2"]
  }
  func.func @missing_payload_block(
      %state: !phys.resource_payload<@ccz_state>,
      %left: !phys.state<@left0>, %right: !phys.state<@right0>) {
    // expected-error @+1 {{payload logical maps must cover every required payload block}}
    %next:2 = phys.unpack_resource %state into(%left, %right) {
      encoding = @toric_encoding,
      payload_action = #qlx.action<ccz>,
      payload_carrier_segments = array<i64: 0, 1, 2>,
      payload_logical_block_ids = ["left_block", "left_block", "left_block"],
      payload_logical_blocks = array<i64: 0, 0, 0>,
      payload_logical_ports = array<i64: 0, 1, 2>
    } : (!phys.resource_payload<@ccz_state>,
         !phys.state<@left0>, !phys.state<@right0>)
      -> (!phys.state<@left0>, !phys.state<@right0>)
    return
  }
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  fabric.code @Toric {
    distance = 1 : i64, n = 1 : i64, partitions = {data = 1 : i64}
  }
  fabric.encoding @toric_encoding {
    code = @Toric, profile = @toric_profile, block = "data",
    logical_ports = ["q0", "q1"]
  }
  func.func @aliased_payload_port(
      %state: !phys.resource_payload<@ccz_state>,
      %carrier: !phys.state<@left0>) {
    // expected-error @+1 {{payload logical mapping must not alias a payload port}}
    %next = phys.unpack_resource %state into(%carrier) {
      encoding = @toric_encoding,
      payload_action = #qlx.action<ccz>,
      payload_carrier_segments = array<i64: 0, 1>,
      payload_logical_block_ids = ["carrier_block", "carrier_block", "carrier_block"],
      payload_logical_blocks = array<i64: 0, 0, 0>,
      payload_logical_ports = array<i64: 0, 0, 1>
    } : (!phys.resource_payload<@ccz_state>, !phys.state<@left0>)
      -> !phys.state<@left0>
    return
  }
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  fabric.code @Toric {
    distance = 1 : i64, n = 1 : i64, partitions = {data = 1 : i64}
  }
  fabric.encoding @toric_encoding {
    code = @Toric, profile = @toric_profile, block = "data",
    logical_ports = ["q0", "q1"]
  }
  func.func @port_exceeds_encoding(
      %state: !phys.resource_payload<@ccz_state>,
      %carrier: !phys.state<@left0>) {
    // expected-error @+1 {{payload logical port index exceeds the encoding logical capacity}}
    %next = phys.unpack_resource %state into(%carrier) {
      encoding = @toric_encoding,
      payload_action = #qlx.action<ccz>,
      payload_carrier_segments = array<i64: 0, 1>,
      payload_logical_block_ids = ["carrier_block", "carrier_block", "carrier_block"],
      payload_logical_blocks = array<i64: 0, 0, 0>,
      payload_logical_ports = array<i64: 0, 1, 2>
    } : (!phys.resource_payload<@ccz_state>, !phys.state<@left0>)
      -> !phys.state<@left0>
    return
  }
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  fabric.code @wide {
    distance = 1 : i64, n = 2 : i64, partitions = {data = 2 : i64}
  }
  fabric.encoding @wide_encoding {
    code = @wide, profile = @wide_profile, block = "data",
    logical_ports = ["q0", "q1", "q2"]
  }
  func.func @unequal_payload_widths(
      %state: !phys.resource_payload<@ccz_state>,
      %a: !phys.state<@a>, %b: !phys.state<@b>,
      %c: !phys.state<@c>, %d: !phys.state<@d>) {
    // expected-error @+1 {{every payload segment width must match its resolved physical resource granularity}}
    %next:4 = phys.unpack_resource %state into(%a, %b, %c, %d) {
      encoding = @wide_encoding,
      payload_action = #qlx.action<ccz>,
      payload_carrier_segments = array<i64: 0, 1, 4>,
      payload_logical_block_ids = ["left_block", "left_block", "right_block"],
      payload_logical_blocks = array<i64: 0, 0, 1>,
      payload_logical_ports = array<i64: 0, 1, 0>
    } : (!phys.resource_payload<@ccz_state>,
         !phys.state<@a>, !phys.state<@b>, !phys.state<@c>, !phys.state<@d>)
      -> (!phys.state<@a>, !phys.state<@b>, !phys.state<@c>, !phys.state<@d>)
    return
  }
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  func.func @missing_encoding(
      %state: !phys.resource_payload<@t_state>,
      %carrier: !phys.state<@carrier>) {
    // expected-error @+1 {{encoding must resolve to fabric.encoding}}
    %next = phys.unpack_resource %state into(%carrier) {
      encoding = @missing_encoding,
      payload_carrier_segments = array<i64: 0, 1>
    } : (!phys.resource_payload<@t_state>, !phys.state<@carrier>)
      -> !phys.state<@carrier>
    return
  }
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  fabric.encoding @missing_code_encoding {
    code = @missing_code, profile = @missing_profile, block = "data",
    logical_ports = ["q0"]
  }
  func.func @missing_code(
      %state: !phys.resource_payload<@t_state>,
      %carrier: !phys.state<@carrier>) {
    // expected-error @+1 {{encoding code must resolve unambiguously to fabric.code}}
    %next = phys.unpack_resource %state into(%carrier) {
      encoding = @missing_code_encoding,
      payload_carrier_segments = array<i64: 0, 1>
    } : (!phys.resource_payload<@t_state>, !phys.state<@carrier>)
      -> !phys.state<@carrier>
    return
  }
}
