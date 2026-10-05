// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module attributes {qlx.profiles = ["p2n"]} {
  fabric.protocol @bad : () -> !fabric.resource<@dog_state> {
    // expected-error @+1 {{resource_kind must match the symbolic result resource type}}
    %state = fabric.produce_resource {
      region = @factory, resource_kind = @cat_state
    } : !fabric.resource<@dog_state>
    fabric.protocol_return %state : !fabric.resource<@dog_state>
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  module @library {
    qlx.action @produce_t : () -> !fabric.resource<@t_state> {
      kind = "produce_t_state"
    }
  }
  // expected-error @+1 {{result types must match the production objective}}
  fabric.protocol @wrong_nested_resource : () -> !fabric.resource<@y_state> attributes {
    objective = @library::@produce_t
  } {
    %state = fabric.produce_resource {
      region = @factory, resource_kind = @y_state
    } : !fabric.resource<@y_state>
    fabric.protocol_return %state : !fabric.resource<@y_state>
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  qlx.action @make_t : () -> !fabric.resource<@t_state> {
    kind = "make_t"
  }
  // expected-error @+1 {{result types must match the production objective}}
  fabric.protocol @wrong_aliased_resource : () -> !fabric.resource<@y_state> attributes {
    objective = @make_t
  } {
    %state = fabric.produce_resource {
      region = @factory, resource_kind = @y_state
    } : !fabric.resource<@y_state>
    fabric.protocol_return %state : !fabric.resource<@y_state>
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  qlx.action @produce_t : () -> !fabric.resource<@t_state> {
    kind = "produce_t_state"
  }
  // expected-error @+1 {{result types must match the production objective}}
  fabric.protocol @wrong_resource : () -> !fabric.resource<@y_state> attributes {
    objective = @produce_t
  } {
    %state = fabric.produce_resource {
      region = @factory, resource_kind = @y_state
    } : !fabric.resource<@y_state>
    fabric.protocol_return %state : !fabric.resource<@y_state>
  }
}
