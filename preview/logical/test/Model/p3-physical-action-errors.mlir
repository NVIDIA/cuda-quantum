// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  // expected-error @+1 {{arity must be positive}}
  phys.action @bad {arity = 0 : i64, process = "{\22kind\22:\22builtin\22,\22name\22:\22h\22,\22parameters\22:{}}"}
}

// -----

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @q {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.graph @bad on @arch : () -> () {
    %0 = phys.acquire [@q0] {event_id = "acquire"} : !phys.state<@q0>
    // expected-error @+1 {{state operand 0 carrier @q0 lacks typed native Pauli-product-rotation capability 'qlx.physical/native_pauli_product_rotation'}}
    %1 = phys.rotate_product %0 {
      angle = 1.250000e-01 : f64, event_id = "rpp", paulis = ["Z"]
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %1 {event_id = "release"} : !phys.state<@q0>
    phys.return
  }
}

// -----

module {
  phys.action @cx {arity = 2 : i64, process = "{\22kind\22:\22builtin\22,\22name\22:\22cx\22,\22parameters\22:{}}"}
  phys.machine @line_arch {
    phys.resource_class @q {
      kind = "qubit", count = 3 : i64, native_actions = [@cx]
    }
    phys.topology @line {
      kind = "explicit", num_nodes = 3 : i64,
      edges = [array<i64: 0, 1>],
      strict
    }
  }
  phys.resource @q0 {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.resource @q2 {kind = "qubit", resource_class = @q, index = 2 : i64}
  phys.graph @bad on @line_arch :
      (!phys.state<@q0>, !phys.state<@q2>) -> () {
  ^bb0(%q0: !phys.state<@q0>, %q2: !phys.state<@q2>):
    // expected-error @+1 {{nonlocal action @cx lane 0 between topology nodes 0 and 2}}
    %0, %1 = phys.apply @cx(%q0, %q2) {
      resources = [@q0, @q2], topology = @line
    } : (!phys.state<@q0>, !phys.state<@q2>) ->
        (!phys.state<@q0>, !phys.state<@q2>)
    phys.release %0, %1 : !phys.state<@q0>, !phys.state<@q2>
    phys.return
  }
}

// -----

module {
  // expected-error @+1 {{controller binding values must be nonempty strings}}
  phys.action @bad {arity = 1 : i64, process = "{}", controller_bindings = {lanes = 1 : i64}}
}

// -----

// A controller binding entry with an empty value is not legal.
module {
  // expected-error @+1 {{controller binding values must be nonempty strings}}
  phys.action @bad {arity = 1 : i64, process = "{}", controller_bindings = {lanes = ""}}
}

// -----

module {
  // expected-error @+1 {{contains duplicate parameter entry 'theta'}}
  phys.action @bad {
    arity = 1 : i64,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22h\22,\22parameters\22:{}}",
    parameters = ["theta", "theta"]
  }
}

// -----

module {
  func.func private @not_an_action()
  phys.machine @bad {
    // expected-error @+1 {{native action @not_an_action must resolve to phys.action}}
    phys.resource_class @q {
      kind = "qubit", count = 1 : i64, native_actions = [@not_an_action]
    }
  }
}

// -----

module {
  phys.action @binary {arity = 2 : i64, process = "{\22kind\22:\22builtin\22,\22name\22:\22cz\22,\22parameters\22:{}}"}
  phys.machine @arch {
    phys.resource_class @q {
      kind = "qubit", count = 1 : i64, native_actions = [@binary]
    }
  }
  phys.resource @q0 {kind = "qubit", resource_class = @q, index = 0 : i64}
  func.func @bad(%arg0: !phys.state<@q0>) {
    // expected-error @+1 {{operand count does not match action @binary arity 2}}
    %0 = phys.apply @binary(%arg0)
      : (!phys.state<@q0>) -> !phys.state<@q0>
    return
  }
}

// -----

module {
  phys.action @h {arity = 1 : i64, process = "{\22kind\22:\22builtin\22,\22name\22:\22h\22,\22parameters\22:{}}", controller_bindings = {lanes = "h"}}
  phys.machine @a {
    phys.resource_class @atom {
      kind = "atom", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @a0 {kind = "atom", resource_class = @atom, index = 0 : i64}
  phys.graph @bad on @a : (!phys.state<@a0>) -> () {
  ^bb0(%arg0: !phys.state<@a0>):
    // expected-error @+1 {{resource class @atom of state operand 0 does not advertise native action @h}}
    %0 = phys.apply @h(%arg0) {resources = [@a0]}
      : (!phys.state<@a0>) -> !phys.state<@a0>
    phys.release %0 : !phys.state<@a0>
    phys.return
  }
}

// -----

module {
  // expected-error @+1 {{broadcast actions must have arity one}}
  phys.action @bad {
    arity = 2 : i64, broadcast, process = "{}", controller_bindings = {lanes = "h"}
  }
}

// -----

module {
  // expected-error @+1 {{modalities was removed; compatibility comes from resource classes and their native_actions}}
  phys.action @bad {
    arity = 1 : i64,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22h\22,\22parameters\22:{}}",
    modalities = ["neutral_atom"]
  }
}

// -----

module {
  // expected-error @+1 {{modality was removed; declare resource kinds, native_actions, native_instruments, and topology instead}}
  phys.machine @bad attributes {modality = "simulator"} {
  }
}

// -----

module {
  phys.machine @a {
    phys.resource_class @q {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.graph @g on @a : () -> () {
    phys.return
  }
  // expected-error @+1 {{initial and final mappings must contain the same carrier roles}}
  phys.mapping @bad {
    graph = @g, source_graph = @source,
    initial = [{role = "patch0.data[0]", resource = @q0}], final = []
  }
}

// -----

module {
  phys.action @swap {arity = 2 : i64, process = "{\22kind\22:\22builtin\22,\22name\22:\22swap\22,\22parameters\22:{}}"}
  phys.action @cx {arity = 2 : i64, process = "{\22kind\22:\22builtin\22,\22name\22:\22cx\22,\22parameters\22:{}}"}
  phys.machine @a {
    phys.resource_class @q {
      kind = "qubit", count = 3 : i64, native_actions = [@swap, @cx]
    }
    phys.topology @line {
      kind = "explicit", num_nodes = 3 : i64,
      edges = [
        array<i64: 0, 1>,
        array<i64: 1, 2>
      ], strict
    }
  }
  phys.graph @g on @a : () -> () {
    phys.return
  }
  // expected-error @+1 {{requires a missing adjacency edge 0 -> 2}}
  phys.routing @bad {
    graph = @g, topology = @line,
    steps = [{event = "route0", action = "cx", path = array<i64: 0, 2, 1>}]
  }
}

// -----

module {
  phys.instrument @wrong_mx {
    kind = "measure_x",
    arity = 1 : i64,
    record_schema = "bit",
    preserves_inputs,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22measure_z\22,\22parameters\22:{}}"
  }
  phys.machine @arch {
    phys.resource_class @q {
      kind = "qubit", count = 1 : i64, native_actions = [],
      native_instruments = [@wrong_mx]
    }
  }
  phys.resource @q0 {
    kind = "qubit", resource_class = @q, index = 0 : i64
  }
  func.func @bad(%arg0: !phys.state<@q0>) {
    // expected-error @+1 {{instrument process must be builtin measure_x for kind measure_x}}
    %0:2 = phys.measure @wrong_mx(%arg0) {record_id = "x"}
      : (!phys.state<@q0>) -> (!phys.state<@q0>, !phys.record<@bit>)
    return
  }
}
